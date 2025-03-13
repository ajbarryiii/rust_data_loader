// src/lib.rs
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use pyo3::exceptions::PyRuntimeError;
use crossbeam::channel::{bounded, unbounded};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use rayon::prelude::*;
use opencv as cv;
use opencv::prelude::*;
use opencv::videoio;
use std::path::{Path, PathBuf};
use std::collections::VecDeque;
use std::time::Instant;

// Custom error type for video loading
#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("Failed to open video file: {0}")]
    VideoOpenError(String),
    
    #[error("Failed to read video frame: {0}")]
    FrameReadError(String),
    
    #[error("Failed to preprocess frame: {0}")]
    PreprocessError(String),
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    
    #[error("Thread error: {0}")]
    ThreadError(String),
}

// Video frame with metadata
#[derive(Clone)]
struct VideoFrame {
    data: Arc<Vec<u8>>,
    width: i32,
    height: i32,
    channels: i32,
    video_idx: usize,
    frame_idx: i64,
    is_padding: bool,  // Flag to indicate if this is a padding frame
}

// Video metadata
#[derive(Clone, Serialize, Deserialize)]
struct VideoMetadata {
    path: PathBuf,
    frame_count: i64,
    fps: f64,
    width: i32,
    height: i32,
    index: usize,
}

// Configuration for the dataloader
#[derive(Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    batch_size: usize,
    num_workers: usize,
    prefetch_factor: usize,
    frame_stride: i64,
    clip_length: i64,
    resize_height: i32,
    resize_width: i32,
    mean: Vec<f32>,
    std: Vec<f32>,
    device: String,
    sampling_mode: SamplingMode,
    clip_overlap: f32,  // Overlap between consecutive clips, as a fraction [0.0, 1.0)
    pad_last: bool,     // Whether to pad the last clip if it's incomplete
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SamplingMode {
    /// Extract clips sequentially with possible overlap
    Sequential,
    /// Extract clips at uniformly spaced intervals
    Uniform,
    /// Extract clips at random positions
    Random,
    /// Use all frames and organize them into overlapping clips
    Dense,
}

impl Default for SamplingMode {
    fn default() -> Self {
        SamplingMode::Sequential
    }
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 4,
            prefetch_factor: 2,
            frame_stride: 1,
            clip_length: 16,
            resize_height: 224,
            resize_width: 224,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            device: "cuda:0".to_string(),
            sampling_mode: SamplingMode::default(),
            clip_overlap: 0.0,
            pad_last: true,
        }
    }
}

// Frame transformation function type
type FrameTransform = Arc<dyn Fn(&Mat) -> Result<Mat, DataLoaderError> + Send + Sync>;

// Main dataloader struct
struct VideoDataLoader {
    videos: Vec<VideoMetadata>,
    config: DataLoaderConfig,
    current_indices: Vec<(usize, i64)>, // (video_idx, frame_idx)
    prefetch_queue: Arc<Mutex<VecDeque<Vec<VideoFrame>>>>,
    worker_handles: Vec<thread::JoinHandle<()>>,
    stop_signal: Arc<Mutex<bool>>,
    transforms: Vec<FrameTransform>,
    runtime: Runtime,
}

impl VideoDataLoader {
    // Constructor
    fn new(
        video_paths: Vec<String>, 
        config: DataLoaderConfig,
        transforms: Vec<FrameTransform>
    ) -> Result<Self, DataLoaderError> {
        // Create tokio runtime for async operations
        let runtime = Runtime::new()
            .map_err(|e| DataLoaderError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other, 
                format!("Failed to create tokio runtime: {}", e)
            )))?;
        
        // Load video metadata in parallel
        let videos = video_paths.par_iter().enumerate().map(|(idx, path)| {
            Self::load_video_metadata(path, idx)
        }).collect::<Result<Vec<_>, _>>()?;
        
        // Calculate initial indices
        let current_indices = videos.iter().map(|v| (v.index, 0i64)).collect();
        
        // Create communication channels
        let prefetch_queue = Arc::new(Mutex::new(VecDeque::new()));
        let stop_signal = Arc::new(Mutex::new(false));
        
        let mut loader = Self {
            videos,
            config,
            current_indices,
            prefetch_queue,
            worker_handles: Vec::new(),
            stop_signal,
            transforms,
            runtime,
        };
        
        // Start worker threads
        loader.start_workers()?;
        
        Ok(loader)
    }
    
    // Load video metadata
    fn load_video_metadata(
        path: &str, 
        index: usize
    ) -> Result<VideoMetadata, DataLoaderError> {
        let cap = videoio::VideoCapture::from_file(path, videoio::CAP_ANY)
            .map_err(|e| DataLoaderError::VideoOpenError(format!("{}: {}", path, e)))?;
        
        let frame_count = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as i64;
        let fps = cap.get(videoio::CAP_PROP_FPS)?;
        let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
        let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
        
        Ok(VideoMetadata {
            path: PathBuf::from(path),
            frame_count,
            fps,
            width,
            height,
            index,
        })
    }
    
    // Start worker threads
    fn start_workers(&mut self) -> Result<(), DataLoaderError> {
        let num_workers = self.config.num_workers;
        let prefetch_count = self.config.prefetch_factor * self.config.batch_size;
        
        let (sender, receiver) = bounded(prefetch_count);
        
        // Spawn worker threads
        for worker_id in 0..num_workers {
            let videos = self.videos.clone();
            let config = self.config.clone();
            let sender = sender.clone();
            let stop_signal = Arc::clone(&self.stop_signal);
            let transforms = self.transforms.clone();
            
            let handle = thread::Builder::new()
                .name(format!("video_loader_{}", worker_id))
                .spawn(move || {
                    Self::worker_loop(
                        worker_id, 
                        videos, 
                        config, 
                        sender, 
                        stop_signal,
                        transforms
                    );
                })
                .map_err(|e| DataLoaderError::ThreadError(format!("Failed to spawn thread: {}", e)))?;
                
            self.worker_handles.push(handle);
        }
        
        // Spawn collector thread
        let prefetch_queue = Arc::clone(&self.prefetch_queue);
        let stop_signal = Arc::clone(&self.stop_signal);
        let config = self.config.clone();
        
        let collector_handle = thread::Builder::new()
            .name("collector".to_string())
            .spawn(move || {
                Self::collector_loop(receiver, prefetch_queue, stop_signal, config.batch_size);
            })
            .map_err(|e| DataLoaderError::ThreadError(format!("Failed to spawn collector thread: {}", e)))?;
            
        self.worker_handles.push(collector_handle);
        
        Ok(())
    }
    
    // Worker thread function
    fn worker_loop(
        worker_id: usize,
        videos: Vec<VideoMetadata>,
        config: DataLoaderConfig,
        sender: crossbeam::channel::Sender<VideoFrame>,
        stop_signal: Arc<Mutex<bool>>,
        transforms: Vec<FrameTransform>,
    ) {
        use rand::prelude::*;
        
        // Assign videos to workers in a round-robin fashion
        let worker_videos: Vec<_> = videos.into_iter()
            .enumerate()
            .filter(|(idx, _)| idx % config.num_workers == worker_id)
            .map(|(_, v)| v)
            .collect();
        
        let mut rng = rand::thread_rng();
            
        // Process assigned videos
        for video in worker_videos {
            let path = video.path.to_str().unwrap_or_default();
            
            // Calculate clip positions based on sampling mode
            let clip_positions = match config.sampling_mode {
                SamplingMode::Sequential => {
                    Self::generate_sequential_clips(
                        video.frame_count, 
                        config.clip_length, 
                        config.frame_stride,
                        config.clip_overlap,
                        config.pad_last
                    )
                },
                SamplingMode::Uniform => {
                    Self::generate_uniform_clips(
                        video.frame_count, 
                        config.clip_length, 
                        config.frame_stride,
                        config.batch_size
                    )
                },
                SamplingMode::Random => {
                    Self::generate_random_clips(
                        video.frame_count, 
                        config.clip_length, 
                        config.frame_stride,
                        config.batch_size,
                        &mut rng
                    )
                },
                SamplingMode::Dense => {
                    Self::generate_dense_clips(
                        video.frame_count, 
                        config.clip_length, 
                        config.frame_stride,
                        config.clip_overlap
                    )
                },
            };
            
            for clip_info in clip_positions {
                // Check stop signal
                if *stop_signal.lock().unwrap() {
                    break;
                }
                
                // Process this clip
                Self::process_clip(
                    &clip_info,
                    &video,
                    &config,
                    &transforms,
                    &sender,
                    worker_id
                );
            }
        }
    }
    
    // Generate sequential clip positions with optional overlap
    fn generate_sequential_clips(
        frame_count: i64,
        clip_length: i64,
        frame_stride: i64,
        overlap: f32,
        pad_last: bool
    ) -> Vec<ClipInfo> {
        let mut clips = Vec::new();
        let stride = (clip_length as f32 * (1.0 - overlap)) as i64;
        let stride = std::cmp::max(1, stride); // Ensure stride is at least 1
        
        let mut start_frame = 0;
        while start_frame < frame_count {
            let remaining = frame_count - start_frame;
            let actual_length = if remaining < clip_length && pad_last {
                clip_length  // We'll pad with duplicated frames
            } else if remaining < clip_length {
                remaining    // We'll use a shorter clip without padding
            } else {
                clip_length  // Normal case: full clip_length
            };
            
            clips.push(ClipInfo {
                start_frame,
                clip_length: actual_length,
                stride: frame_stride,
                needs_padding: remaining < clip_length && pad_last,
            });
            
            start_frame += stride;
        }
        
        clips
    }
    
    // Generate uniformly spaced clips
    fn generate_uniform_clips(
        frame_count: i64,
        clip_length: i64,
        frame_stride: i64,
        num_clips: usize
    ) -> Vec<ClipInfo> {
        let mut clips = Vec::with_capacity(num_clips);
        
        if frame_count <= clip_length {
            // Just one clip for the entire video
            clips.push(ClipInfo {
                start_frame: 0,
                clip_length,
                stride: frame_stride,
                needs_padding: frame_count < clip_length,
            });
            return clips;
        }
        
        // Maximum valid starting position
        let max_start = frame_count - clip_length;
        
        for i in 0..num_clips as i64 {
            let start_frame = if num_clips > 1 {
                (i * max_start) / (num_clips as i64 - 1)
            } else {
                0
            };
            
            clips.push(ClipInfo {
                start_frame,
                clip_length,
                stride: frame_stride,
                needs_padding: false, // No padding needed for uniform sampling
            });
        }
        
        clips
    }
    
    // Generate random clip positions
    fn generate_random_clips(
        frame_count: i64,
        clip_length: i64,
        frame_stride: i64,
        num_clips: usize,
        rng: &mut impl rand::Rng
    ) -> Vec<ClipInfo> {
        let mut clips = Vec::with_capacity(num_clips);
        
        if frame_count <= clip_length {
            // Just one clip for the entire video
            clips.push(ClipInfo {
                start_frame: 0,
                clip_length,
                stride: frame_stride,
                needs_padding: frame_count < clip_length,
            });
            return clips;
        }
        
        // Maximum valid starting position
        let max_start = frame_count - clip_length;
        
        for _ in 0..num_clips {
            let start_frame = rng.gen_range(0..=max_start);
            
            clips.push(ClipInfo {
                start_frame,
                clip_length,
                stride: frame_stride,
                needs_padding: false, // No padding needed for random sampling
            });
        }
        
        clips
    }
    
    // Generate densely overlapping clips
    fn generate_dense_clips(
        frame_count: i64,
        clip_length: i64,
        frame_stride: i64,
        overlap: f32
    ) -> Vec<ClipInfo> {
        let mut clips = Vec::new();
        let stride = (clip_length as f32 * (1.0 - overlap)) as i64;
        let stride = std::cmp::max(1, stride); // Ensure stride is at least 1
        
        let mut start_frame = 0;
        while start_frame <= frame_count - clip_length {
            clips.push(ClipInfo {
                start_frame,
                clip_length,
                stride: frame_stride,
                needs_padding: false,
            });
            
            start_frame += stride;
        }
        
        // Add one final clip if needed and padding is allowed
        if start_frame < frame_count && start_frame > 0 {
            clips.push(ClipInfo {
                start_frame: frame_count - clip_length,
                clip_length,
                stride: frame_stride,
                needs_padding: false,  // No padding needed since we ensure it fits
            });
        }
        
        clips
    }
    
    // Clip information for processing
    #[derive(Clone, Debug)]
    struct ClipInfo {
        start_frame: i64,
        clip_length: i64,
        stride: i64,
        needs_padding: bool,
    }
    
    // Process a single clip and send frames to the collector
    fn process_clip(
        clip_info: &ClipInfo,
        video: &VideoMetadata,
        config: &DataLoaderConfig,
        transforms: &[FrameTransform],
        sender: &crossbeam::channel::Sender<VideoFrame>,
        worker_id: usize,
    ) {
        let path = video.path.to_str().unwrap_or_default();
        
        // Open the video
        let mut cap = match videoio::VideoCapture::from_file(path, videoio::CAP_ANY) {
            Ok(cap) => cap,
            Err(e) => {
                eprintln!("Worker {}: Failed to open video {}: {}", worker_id, path, e);
                return;
            }
        };
        
        // Seek to the starting frame
        if clip_info.start_frame > 0 {
            if let Err(e) = cap.set(videoio::CAP_PROP_POS_FRAMES, clip_info.start_frame as f64) {
                eprintln!("Worker {}: Failed to seek to frame {}: {}", 
                         worker_id, clip_info.start_frame, e);
                return;
            }
        }
        
        // Process frames for this clip
        let mut frames_read = 0;
        let mut last_valid_frame: Option<Mat> = None;
        
        while frames_read < clip_info.clip_length {
            // Read frame
            let mut frame = Mat::default();
            let read_success = match cap.read(&mut frame) {
                Ok(success) => success,
                Err(e) => {
                    eprintln!("Worker {}: Error reading frame: {}", worker_id, e);
                    break;
                }
            };
            
            // Check if we reached the end of the video
            if !read_success {
                // If we need padding and have a previous frame, use it
                if clip_info.needs_padding && last_valid_frame.is_some() {
                    let mut buffer = Vec::new();
                    let last_frame = last_valid_frame.as_ref().unwrap();
                    
                    // Apply transformations to the last valid frame
                    let processed_frame = match Self::apply_transforms(last_frame, transforms) {
                        Ok(f) => f,
                        Err(e) => {
                            eprintln!("Worker {}: Error processing padding frame: {}", worker_id, e);
                            break;
                        }
                    };
                    
                    // Convert to bytes
                    let step = processed_frame.step1(0).unwrap_or(0) as usize;
                    let data_slice = unsafe { 
                        std::slice::from_raw_parts(
                            processed_frame.data() as *const u8, 
                            step * processed_frame.rows() as usize
                        ) 
                    };
                    buffer.extend_from_slice(data_slice);
                    
                    // Create frame object (marked as padding)
                    let video_frame = VideoFrame {
                        data: Arc::new(buffer),
                        width: processed_frame.cols(),
                        height: processed_frame.rows(),
                        channels: processed_frame.channels(),
                        video_idx: video.index,
                        frame_idx: clip_info.start_frame + frames_read,
                        is_padding: true,
                    };
                    
                    // Send to collector
                    if sender.send(video_frame).is_err() {
                        break;
                    }
                    
                    frames_read += 1;
                    continue;
                } else {
                    // No padding or no valid frame to pad with
                    break;
                }
            }
            
            // Skip frames according to stride if needed
            if frames_read % clip_info.stride != 0 {
                frames_read += 1;
                last_valid_frame = Some(frame);
                continue;
            }
            
            // Remember this frame for potential padding
            last_valid_frame = Some(frame.clone());
            
            // Apply transformations
            let processed_frame = match Self::apply_transforms(&frame, transforms) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Worker {}: Error processing frame: {}", worker_id, e);
                    frames_read += 1;
                    continue;
                }
            };
            
            // Convert to bytes
            let mut buffer = Vec::new();
            let step = processed_frame.step1(0).unwrap_or(0) as usize;
            let data_slice = unsafe { 
                std::slice::from_raw_parts(
                    processed_frame.data() as *const u8, 
                    step * processed_frame.rows() as usize
                ) 
            };
            buffer.extend_from_slice(data_slice);
            
            // Create frame object
            let video_frame = VideoFrame {
                data: Arc::new(buffer),
                width: processed_frame.cols(),
                height: processed_frame.rows(),
                channels: processed_frame.channels(),
                video_idx: video.index,
                frame_idx: clip_info.start_frame + frames_read,
                is_padding: false,
            };
            
            // Send to collector
            if sender.send(video_frame).is_err() {
                break;
            }
            
            frames_read += 1;
        }
    }
    
    // Apply frame transformations
    fn apply_transforms(
        frame: &Mat, 
        transforms: &[FrameTransform]
    ) -> Result<Mat, DataLoaderError> {
        let mut current = frame.clone();
        
        for transform in transforms {
            current = transform(&current)?;
        }
        
        Ok(current)
    }
    
    // Collector thread function
    fn collector_loop(
        receiver: crossbeam::channel::Receiver<VideoFrame>,
        prefetch_queue: Arc<Mutex<VecDeque<Vec<VideoFrame>>>>,
        stop_signal: Arc<Mutex<bool>>,
        batch_size: usize,
    ) {
        let mut current_batch = Vec::with_capacity(batch_size);
        
        while !*stop_signal.lock().unwrap() {
            match receiver.recv() {
                Ok(frame) => {
                    current_batch.push(frame);
                    
                    if current_batch.len() >= batch_size {
                        let mut queue = prefetch_queue.lock().unwrap();
                        queue.push_back(std::mem::take(&mut current_batch));
                        current_batch = Vec::with_capacity(batch_size);
                    }
                },
                Err(_) => break,
            }
        }
        
        // Push any remaining frames
        if !current_batch.is_empty() {
            let mut queue = prefetch_queue.lock().unwrap();
            queue.push_back(current_batch);
        }
    }
    
    // Get next batch
    fn next_batch(&mut self) -> Option<Vec<VideoFrame>> {
        let mut queue = self.prefetch_queue.lock().unwrap();
        queue.pop_front()
    }
    
    // Clean up resources
    fn cleanup(&mut self) {
        // Set stop signal
        {
            let mut signal = self.stop_signal.lock().unwrap();
            *signal = true;
        }
        
        // Join worker threads
        while let Some(handle) = self.worker_handles.pop() {
            let _ = handle.join();
        }
    }
}

// Default transforms
fn create_default_transforms(config: &DataLoaderConfig) -> Vec<FrameTransform> {
    let resize_height = config.resize_height;
    let resize_width = config.resize_width;
    let mean = config.mean.clone();
    let std = config.std.clone();
    
    vec![
        // Resize transform
        Arc::new(move |frame: &Mat| -> Result<Mat, DataLoaderError> {
            let mut resized = Mat::default();
            cv::imgproc::resize(
                frame,
                &mut resized,
                cv::core::Size::new(resize_width, resize_height),
                0.0,
                0.0,
                cv::imgproc::INTER_LINEAR,
            ).map_err(|e| DataLoaderError::PreprocessError(format!("Resize error: {}", e)))?;
            Ok(resized)
        }),
        
        // Normalization transform
        Arc::new(move |frame: &Mat| -> Result<Mat, DataLoaderError> {
            let mut normalized = frame.clone();
            normalized.convert_to(
                &mut normalized, 
                cv::core::CV_32F, 
                1.0/255.0, 
                0.0
            ).map_err(|e| DataLoaderError::PreprocessError(format!("Convert error: {}", e)))?;
            
            let mean_scalar = cv::core::Scalar::new(mean[0] as f64, mean[1] as f64, mean[2] as f64, 0.0);
            let std_scalar = cv::core::Scalar::new(1.0/std[0] as f64, 1.0/std[1] as f64, 1.0/std[2] as f64, 1.0);
            
            cv::core::subtract(&normalized, &mean_scalar, &mut normalized, &cv::core::no_array(), -1)
                .map_err(|e| DataLoaderError::PreprocessError(format!("Subtract error: {}", e)))?;
                
            cv::core::multiply(&normalized, &std_scalar, &mut normalized, 1.0, -1)
                .map_err(|e| DataLoaderError::PreprocessError(format!("Multiply error: {}", e)))?;
                
            Ok(normalized)
        }),
    ]
}

// Python module
#[pymodule]
fn rust_video_loader(_py: Python, m: &PyModule) -> PyResult<()> {
    // Python wrapper for VideoDataLoader
    #[pyclass]
    struct PyVideoDataLoader {
        inner: Option<VideoDataLoader>,
        config: DataLoaderConfig,
    }
    
    #[pymethods]
    impl PyVideoDataLoader {
        #[new]
        fn new(
            video_paths: Vec<String>,
            batch_size: Option<usize>,
            num_workers: Option<usize>,
            prefetch_factor: Option<usize>,
            frame_stride: Option<i64>,
            clip_length: Option<i64>,
            resize_height: Option<i32>,
            resize_width: Option<i32>,
            mean: Option<Vec<f32>>,
            std: Option<Vec<f32>>,
            device: Option<String>,
            sampling_mode: Option<String>,
            clip_overlap: Option<f32>,
            pad_last: Option<bool>,
        ) -> PyResult<Self> {
            let mut config = DataLoaderConfig::default();
            
            if let Some(val) = batch_size { config.batch_size = val; }
            if let Some(val) = num_workers { config.num_workers = val; }
            if let Some(val) = prefetch_factor { config.prefetch_factor = val; }
            if let Some(val) = frame_stride { config.frame_stride = val; }
            if let Some(val) = clip_length { config.clip_length = val; }
            if let Some(val) = resize_height { config.resize_height = val; }
            if let Some(val) = resize_width { config.resize_width = val; }
            if let Some(val) = mean { config.mean = val; }
            if let Some(val) = std { config.std = val; }
            if let Some(val) = device { config.device = val; }
            if let Some(val) = clip_overlap { 
                config.clip_overlap = val.clamp(0.0, 0.99); 
            }
            if let Some(val) = pad_last { config.pad_last = val; }
            
            // Parse sampling mode string
            if let Some(mode_str) = sampling_mode {
                config.sampling_mode = match mode_str.to_lowercase().as_str() {
                    "sequential" => SamplingMode::Sequential,
                    "uniform" => SamplingMode::Uniform,
                    "random" => SamplingMode::Random,
                    "dense" => SamplingMode::Dense,
                    _ => return Err(PyRuntimeError::new_err(
                        format!("Invalid sampling mode: '{}'. Valid options are: 'sequential', 'uniform', 'random', 'dense'", mode_str)
                    )),
                };
            }
            
            let transforms = create_default_transforms(&config);
            
            let loader = VideoDataLoader::new(video_paths, config.clone(), transforms)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create dataloader: {}", e)))?;
                
            Ok(Self {
                inner: Some(loader),
                config,
            })
        }
        
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }
        
        fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<PyObject>> {
            if let Some(loader) = &mut slf.inner {
                if let Some(batch) = loader.next_batch() {
                    // Convert batch to PyTorch tensors
                    let result = Self::convert_batch_to_pytorch(py, batch, &slf.config)?;
                    return Ok(Some(result));
                }
            }
            
            Ok(None)
        }
        
        fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }
        
        fn __exit__(&mut self, _exc_type: &PyAny, _exc_value: &PyAny, _traceback: &PyAny) -> PyResult<bool> {
            if let Some(loader) = &mut self.inner {
                loader.cleanup();
            }
            self.inner = None;
            Ok(true)
        }
        
        // Method to get a single batch
        fn get_batch(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
            if let Some(loader) = &mut self.inner {
                if let Some(batch) = loader.next_batch() {
                    let result = Self::convert_batch_to_pytorch(py, batch, &self.config)?;
                    return Ok(Some(result));
                }
            }
            
            Ok(None)
        }
    }
    
    impl PyVideoDataLoader {
        // Helper method to convert batch to PyTorch format
        fn convert_batch_to_pytorch(
            py: Python<'_>,
            batch: Vec<VideoFrame>,
            config: &DataLoaderConfig,
        ) -> PyResult<PyObject> {
            // Import torch
            let torch = PyModule::import(py, "torch")?;
            
            // Create list for frame data
            let py_frames = PyList::empty(py);
            let py_video_indices = PyList::empty(py);
            let py_frame_indices = PyList::empty(py);
            
            for frame in batch {
                // Convert frame data to Python bytes
                let data_bytes = PyBytes::new(py, &frame.data);
                py_frames.append(data_bytes)?;
                
                // Add metadata
                py_video_indices.append(frame.video_idx)?;
                py_frame_indices.append(frame.frame_idx)?;
            }
            
            // Convert to numpy array then to torch tensor
            let np = PyModule::import(py, "numpy")?;
            let h = config.resize_height;
            let w = config.resize_width;
            let c = 3; // Assuming RGB
            
            let np_array = np.getattr("frombuffer")?.call1((
                py_frames,
                np.getattr("uint8")?,
            ))?;
            
            let reshaped = np_array.call_method1(
                "reshape", 
                (batch.len(), h, w, c)
            )?;
            
            // Convert to torch tensor and change layout to NCHW
            let tensor = torch.getattr("from_numpy")?.call1((reshaped,))?;
            let permuted = tensor.call_method1("permute", (0, 3, 1, 2))?;
            let float_tensor = permuted.call_method0("float")?;
            
            // Create dictionary with batch data
            let locals = PyDict::new(py);
            locals.set_item("frames", float_tensor)?;
            locals.set_item("video_indices", py_video_indices)?;
            locals.set_item("frame_indices", py_frame_indices)?;
            
            // Create and return dictionary
            let result = py.eval("{'frames': frames, 'video_indices': video_indices, 'frame_indices': frame_indices}", None, Some(locals))?;
            
            Ok(result.to_object(py))
        }
    }
    
    m.add_class::<PyVideoDataLoader>()?;
    
    Ok(())
}
