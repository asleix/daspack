use anyhow::Result;
use ndarray::{Array2, ArrayView2};

use crate::codec::CodecParams;

use super::{Codec, Quantizer, CodecError};


/// Adapter/Decorator that wraps a lossless Codec and a Quantizer
pub struct LossyCodec<C: Codec, Q: Quantizer> {
    inner: C,
    quantizer: Q,
}

impl<C: Codec, Q: Quantizer> LossyCodec<C, Q> {
    /// Create a new lossy codec from a lossless codec and a quantizer
    pub fn new(inner: C, quantizer: Q) -> Self {
        LossyCodec { inner, quantizer }
    }
}

impl<C, Q> Codec for LossyCodec<C, Q>
where
    C: Codec<SourceType = i32>,              
    Q: Quantizer + Clone,     
{
    type SourceType = Q::SourceType;
    fn compress(&self, data: ArrayView2<Q::SourceType>) -> Result<Vec<u8>> {
        // 1) quantize floats -> ints
        let ints = self.quantizer.quantize(data);
        // 2) delegate to the lossless compressor
        self.inner.compress(ints.view())
    }

    fn decompress(&self, stream: &[u8], shape: (usize, usize)) -> Result<Array2<Q::SourceType>> {
        // 1) decompress back to ints
        let ints = self.inner.decompress(stream, shape)?;
        // 2) dequantize ints -> floats
        Ok(self.quantizer.dequantize(ints.view()))
    }
}

/// A simple uniform quantizer implementation
#[derive(Debug, Clone)]
pub struct UniformQuantizer {
    pub step: f32,
}

impl UniformQuantizer {
    /// Create a uniform quantizer with the given step size
    pub fn new(step: f32) -> Self {
        UniformQuantizer { step }
    }
}

impl Quantizer for UniformQuantizer {
    type SourceType = f64;
    fn quantize(&self, data: ArrayView2<f64>) -> Array2<i32> {
        data.mapv(|x| (x / self.step as f64).round() as i32)
    }

    fn dequantize(&self, data: ArrayView2<i32>) -> Array2<f64> {
        data.mapv(|x| x as f64 * self.step as f64)
    }
}

impl CodecParams for UniformQuantizer {
    fn serialize(&self) -> Result<Vec<u8>, CodecError> {
        // we only need to record the step size as an f64
        let mut buf = Vec::with_capacity(4);
        buf.extend(&self.step.to_le_bytes());
        Ok(buf)
    }

    fn read(data: &[u8]) -> Self {
        // expect exactly 8 bytes
        let bytes: [u8; 4] = data
            .try_into()
            .expect("UniformQuantizer::read: expected 4 bytes");
        let step = f32::from_le_bytes(bytes);
        UniformQuantizer::new(step)
    }
}

#[derive(Debug, Clone)]
pub struct LosslessQuantizer;

impl Quantizer for LosslessQuantizer {
    type SourceType = i32;
     fn quantize(&self, data: ArrayView2<i32>) -> Array2<i32> {
        data.to_owned()
    }

    fn dequantize(&self, data: ArrayView2<i32>) -> Array2<i32> {
        data.to_owned()
    }
}

impl CodecParams for LosslessQuantizer {
    fn serialize(&self) -> Result<Vec<u8>, CodecError> {
        let buf = Vec::new();
        Ok(buf)
    }

    fn read(_data: &[u8]) -> Self {
        LosslessQuantizer
    }
}


#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuantKind {
    Uniform  = 1,
    Lossless = 2,
}

impl QuantKind {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(QuantKind::Uniform),
            2 => Some(QuantKind::Lossless),
            _ => None,
        }
    }
}

pub trait QuantMeta {
    const KIND: QuantKind;
}

impl QuantMeta for UniformQuantizer  { const KIND: QuantKind = QuantKind::Uniform; }
impl QuantMeta for LosslessQuantizer { const KIND: QuantKind = QuantKind::Lossless; }


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};
    use std::convert::TryInto;

    /// A simple lossless codec for testing: round-trips i32 arrays to bytes.
    struct IdentityCodec;

    impl Codec for IdentityCodec {
        type SourceType = i32;
        fn compress(&self, data: ArrayView2<i32>) -> Result<Vec<u8>> {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &val in data.iter() {
                buf.extend(&val.to_le_bytes());
            }
            Ok(buf)
        }
        fn decompress(&self, stream: &[u8], shape: (usize, usize)) -> Result<Array2<i32>> {
            let (rows, cols) = shape;
            assert_eq!(stream.len(), rows * cols * 4);
            let mut vec = Vec::with_capacity(rows * cols);
            for chunk in stream.chunks_exact(4) {
                let val = i32::from_le_bytes(chunk.try_into().unwrap());
                vec.push(val);
            }
            Ok(Array2::from_shape_vec((rows, cols), vec).unwrap())
        }
    }

    #[test]
    fn test_uniform_quantizer_roundtrip() {
        let step = 0.5;
        let quant = UniformQuantizer::new(step);
        let data = arr2(&[
            [0.0, 0.25, -1.2],
            [3.3, -4.7, 9.99],
        ]);
        let ints = quant.quantize(data.view());
        let restored = quant.dequantize(ints.view());
        let epsilon = step / 2.0;
        for ((i, j), &orig) in data.indexed_iter() {
            let dec = restored[[i, j]];
            assert!((dec - orig).abs() <= epsilon as f64,
                "At ({},{}): got {}, expected approx {}", i, j, dec, orig);
        }
    }

    #[test]
    fn test_lossy_codec_identity() {
        let step = 1.0;
        let quant = UniformQuantizer::new(step);
        let codec = LossyCodec::new(IdentityCodec, quant);
        let data = arr2(&[
            [1.2, 2.8],
            [-3.7, 4.1],
        ]);
        let compressed = codec.compress(data.view()).unwrap();
        let restored = codec.decompress(&compressed, data.dim()).unwrap();
        let epsilon = step / 2.0;
        for ((i, j), &orig) in data.indexed_iter() {
            let expected = (orig / step as f64).round() * step as f64;
            let dec = restored[[i, j]];
            assert!((dec - expected).abs() <= epsilon as f64,
                "Lossy at ({},{}): got {}, expected approx {}", i, j, dec, expected);
        }
    }

    #[test]
    fn test_compress_decompress_empty() {
        let step = 0.1;
        let quant = UniformQuantizer::new(step);
        let codec = LossyCodec::new(IdentityCodec, quant);
        let data: Array2<f64> = Array2::zeros((0, 0));
        let compressed = codec.compress(data.view()).unwrap();
        assert!(compressed.is_empty());
        let restored = codec.decompress(&compressed, (0, 0)).unwrap();
        assert_eq!(restored.shape(), &[0, 0]);
    }
}