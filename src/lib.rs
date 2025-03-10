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

    #[error("Thread error: {0}")
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
    mean= Vec<f32>,
    std= Vec<f32>,
    device: String,
    clip_overlap: f32,
    pad_last=bool,
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
        path=&str,
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
}


