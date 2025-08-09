
use anyhow::{Result};
use ndarray::{Array2, ArrayView2};


mod blocks;
mod params;
mod common;
mod lossless;
mod lossy;
mod dascoder;

pub use lossless::LosslessCodec;
pub use params::CompressParams;
pub use lossy::{LossyCodec, UniformQuantizer, LosslessQuantizer};
pub use dascoder::{DASCoder, Decoded};


pub trait Codec: Send + Sync {
    type SourceType: Copy + 'static;
    /// Compress the full 2-D data array into one byte-stream.
    fn compress(&self, data: ArrayView2<Self::SourceType>) -> Result<Vec<u8>>;

    /// Decompress `stream` back to an `Array2<i32>` of shape `shape`.
    fn decompress(&self, stream: &[u8], shape: (usize, usize)) -> Result<Array2<Self::SourceType>>;
}

/// Error type for codec serialization failures
#[derive(Debug)]
pub enum CodecError {
    /// Failed integer cast (e.g., length/offset math)
    IntConversion(std::num::TryFromIntError),
    /// Problems with the container/bitstream layout (magic, version, sizes…)
    Format(String),
    /// Anything else bubbled up via anyhow
    Other(anyhow::Error),
}

impl From<std::num::TryFromIntError> for CodecError {
    fn from(err: std::num::TryFromIntError) -> Self {
        CodecError::IntConversion(err)
    }
}

impl From<anyhow::Error> for CodecError {
    fn from(err: anyhow::Error) -> Self {
        CodecError::Other(err)
    }
}

// Quality-of-life: let you write `?` on io ops or use string literals.
impl From<std::io::Error> for CodecError {
    fn from(err: std::io::Error) -> Self {
        CodecError::Other(err.into())
    }
}
impl From<&'static str> for CodecError {
    fn from(msg: &'static str) -> Self {
        CodecError::Format(msg.to_string())
    }
}
impl From<String> for CodecError {
    fn from(msg: String) -> Self {
        CodecError::Format(msg)
    }
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::IntConversion(e) => write!(f, "Integer conversion error: {}", e),
            CodecError::Format(msg)      => write!(f, "Format error: {}", msg),
            CodecError::Other(e)         => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for CodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CodecError::IntConversion(e) => Some(e),
            CodecError::Format(_)        => None,
            CodecError::Other(e)         => Some(e.as_ref()),
        }
    }
}



/// Trait for codec parameters: must be serializable to bytes and readable from bytes
pub trait CodecParams: Send + Sync {
    /// Serialize the parameters into a Vec<u8>, or return an error if a value is out of range
    fn serialize(&self) -> Result<Vec<u8>, CodecError>;
    /// Read (deserialize) the parameters from a byte slice
    fn read(data: &[u8]) -> Self
    where
        Self: Sized;
}

/// Quantizer: maps between f64 data and integer representations
pub trait Quantizer: Send + Sync {
    type SourceType: Copy + 'static;
    /// Convert a view of f64 data into i32 values
    fn quantize(&self, data: ArrayView2<Self::SourceType>) -> Array2<i32>;
    /// Convert a view of i32 data back into f64 values
    fn dequantize(&self, data: ArrayView2<i32>) -> Array2<Self::SourceType>;
}

