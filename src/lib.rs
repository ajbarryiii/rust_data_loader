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
