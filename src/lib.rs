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
