
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
pub use dascoder::DASCoder;


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
    IntConversion(std::num::TryFromIntError),
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

