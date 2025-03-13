use pyo3::prelude::*;
use pyo3::types::{PyBytes,PyList};
use pyo3::exceptions::PyRuntimeError;
use std::sync::{Arc,Mutex};
use std::thread;
use tokio::runtime::Runtime;
use seride::{Serialize,Deserialize};
use thiserror::Error;
use rayon::prelude::*;
use opencv as cv;
use opencv::prelude::*;
use opencv::videoio;
use std::path::{Path,PathBuf};
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Error,Debug)]

pub enum DataLoaderError {
    #[error("Failed to open video file: {0}")]
    VideoOpenError(String),

    #[error("Failed to read video frame: {0}")]
    FrameReadError(String),

    #[error("Failed to read video frame: {0}")]
    PreProcessError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),

    #[error("Thread error: {0}")]
    TreadError(String),
}

#[derive(Clone)]
struct VideoFrame {
    data: Arc<Vec<u8>>,
    width: i32,
    height: i32,
    channels: i32,
    video_idx: usize,
    frame_idx: i64,
    is_padding: bool, // flag to indicate if we had to padd the end of the video
}

#[derive(Clone, Serialize, Deserialize)]
struct MetaData {
    path: PathBuf,
    frame_count: i64,
    fps: f64,
    width: i32,
    height: i32,
    index: usize,
}

#[derive(Clone,Serialize,Deserialize)]
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
    clip_overlap: f32,
    pad_last: bool,
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
            clip_overlap: 0.0,
            pad_last: true,
        }
    }
}

type FrameTransform = Arc<dyn Fn(&Mat)->Result<Mat,DataLoaderError>+Send+Sync>;

struct VideoDataLoader {
    videos: Vec<Metadata>,
    config: DataLoaderConfig,
    current_indices: Vec<(usize,i64)>,
    prefetch_queue: Arc<Mutex<VecDeque<Vec<VideoFrame>>>>,
    worker_handles: Vec<thread::JoinHandle<()>>,
    stop_signal: Arc<Mutex<bool>>,
    transforms: Vec<FrameTransform>,
    runtime: Runtime,
}

impl VideoDataLoader {
    fn new(
        video_paths: Vec<String>,
        config: DataLoaderConfig,
        transforms: Vec<FrameTransform>
    )->Result<Self,DataLoaderError> {
        let runtime=Runtime::new()
            .map_err(|e| DataLoaderError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to create tokio runtime: {}",e)
            )))?;

        let videos=video_paths.par_iter().enumerate().map(|(idx,path)| {
            Self::load_video_metadata(path,idx)
        }).collect::<Result<Vec<_>, _>>()?;

        let current_indices=videos.iter().map(|v| (v.index,0i64)).collect();

        let prefetch_queue=Arc::new(Mutex::new(VecDeque::new()));
        let stop_signal=Arc::new(Mutex::new(false));
        
        let mut loader=Self {
            videos,
            config,
            current_indices,
            prefetch_queue,
            worker_handles: Vec::new(),
            stop_signal,
            transforms,
            runtime,
        };
        loader.start_workers()?;
        Ok(loader)
    }

    fn load_metadata(
        path:&str,
        index: usize
    )->Result<MetaData, DataLoaderError> {
        let cap=videoio::VideoCapture::from_file(path,videoio::CAP_ANY)
            .map_err(|e| DataLoaderError::VideoOpenError(format!("{}:{}", path,e)))?;
        let frame_count=cap.get(videoio::CAP_PROP_FRAME_COUNT)? as i64;
        let fps=cap.get(videoio::CAP_PROP_FPS)?;
        let width=cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
        let height=cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

        Ok(MetaData {
            path: PathBuf::from(path),
            frame_count,
            fps,
            width,
            height,
            index,
        })
    }

    fn start_workers(&mut self) -> Result<(),DataLoaderError> {
        let num_workers=self.config.num_workers;
        let prefetch_count=self.config.prefetch_factor*self.config.batch_size;
        let (sender,receiver)=bounded(prefetch_count);

        for worker_id in 0..num_workers {
            let videos=self.videos.clone();
            let config=self.config.clone();
            let sender=sender.clone();
            let stop_signal=Arc::clone(&self.stop_signal);
            let transforms=self.transforms.clone();

            let handle=thread::Builder::new()
                .name(format!("video_loader_{}",worker_id))
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

        let prefetch_queue=Arc::clone(&self.prefetch_queue);
        let stop_signal=Arc::clone(&self.stop_signal);
        let config=self.config.clone();

        let collector_handle=thread::Builder::new()
            .name("collector".to_string())
            .spawn(move || {
                Self::collector_loop(receiver, prefetch_queue, stop_signal, config.batch_size);
            })
            .map_err(|e| DataLoaderError::ThreadError(format!("Failed to spawn collector thread: {}",e)))?;

        self.worker_handles.push(collector_handles);

        Ok(())
    }

    fn worker_loop(
        worker_id: usize,
        videos: Vec<MetaData>,
        config: DataLoaderConfig,
        sender: crossbeam::channel::Sender<VideoFRame>,
        stop_signal: Arc<Mutex<bool>>,
        transforms: Vec<FrameTransform>,
    ) {
        use rand::prelude::*;

        let worker_videos: Vec<_> = videos.into_iter()
            .enumerate()
            .filter()
            .map(|(_,v)| v)
            .collect();

        let mut rng=rand::thread_rng();
        
        for video in worker_videos {
            let path=video.path.to_str().unwrap_or_default();

            let clip_positions= Self::generate_sequential_clips(
                video.frame_count,
                config.clip_length,
                config.frame_stride,
                config.clip_overlap,
                config.pad_last
                );
            for clip_info in clip_positions {
                if *stop_signal.lock().unwrap() {
                    break;
                }
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

    fn generate_sequential_clips(
        frame_count: i64,
        clip_length: i64,
        frame_stride: i64,
        overlap: f32,
        pad_last: bool
    ) -> Vec<ClipInfo> {
        let mut clips=Vec::new();
        let stride=(clip_length as f32 * (1.0-overlap)) as i64;
        let stride=std::cmp::max(1,stride);

        let mut start_frame=0;
        while start_frame<frame_count {
            let remaining=frame_count-start_frame;;
            let actual_length=if remaining<clip_length&&pad_last {
                clip_length
            } else if remaining < clip_length {
                remaining
            } else {
                clip_length
            };

            clips.push(ClipInfo {
                start_frame,
                clip_length: actual_length,
                stride: frame_stride,
                needs_padding: remaining<clip_length && pad_last,
            });
            start_frame+=stride;
        }
        clips
    }

    #[derive(Clone,Debug)]
    struct ClipInfo {
        start_frame: i64,
        clip_length: i64,
        stride: i64,
        needs_padding: bool,
    }

    fn process_clip(
        clip_info: &ClipInfo,
        video: &Metadata,
        config: &DataLoaderConfig,
        transforms: &[FrameTransform],
        sender: &crossbeam::channel::Sender<VideoFrame>,
        worker_id: usize,
    ) {
        let path=video.path.to_str().unwrap_or_default();
        let mut cap=match videoio::VideoCapture::from_file(path,videoio::CAP_ANY) {
            if clip_info.start_frame>0 {
                if let Err(e) = cap.set(videoio::CAP_PROP_POS_FRAMES, clip_info.start_frame as f64) {
                    eprintln!("Worker {}: Failed to seek to frame {}: {}"'
                        worker_id,clip_info.start_frame,e);
                    return;
                }
            }

            let mut frames_read=0;
            let mut last_valid_frame: Option<Mat> = None;

            while frames_read<clip_info.clip_length {
                let mut frame=Mat::default();
                let read_sucess = match cap.read(&mut frame) {
                    Ok(success) => success,
                    Err(e)=> {
                        eprintln!("Worker {}: Error reading frame: {}", worker_id, e);
                        break;
                    }
                };

                if !read_success {
                    if clip_info.needs_padding && last_valid_frame.is_some() {
                        let mut buffer = Vec::new();
                        let mut last_frame=last_valid_frame.as_ref().unwrap();

                        let processed_frame=match Self::apply_transforms(last_frame,transforms) {
                            Ok(f) =>f,
                            Err(e)=> {
                                eprinln!("Worker {}: Error processing padding frame: {}", worker_id,e);
                                break;
                            }
                        };
                        
                        let step=processed_frame.step1(0).unwrap_or(0) as usize;
                        let data_slice= unsafe {
                            std::slice::from_raw_parts(
                                processed_frame.data() as *const u8,
                                step * processed_frame.rows() as usize
                            )
                        };
                        
                        buffer.extend_from_slice(data_slice);
                    }
                }
            }

        }
    }
}


