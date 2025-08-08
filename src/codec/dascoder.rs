// daspacker.rs

use anyhow::Result;
use ndarray::{Array2, ArrayView2};
use std::{convert::TryInto, marker::PhantomData};

use super::{
    Codec,
    CodecError,
    LosslessCodec,
    LossyCodec,
    CompressParams,
    CodecParams,
    Quantizer,
};

/// High-level packer/unpacker for DASPack bitstreams.
/// 
/// Parameterized on your Quantizer type `Q`, which must
/// implement `Quantizer + CodecParams + Clone`.
pub struct DASCoder<Q>
where
    Q: Quantizer + CodecParams + Clone,
{
    /// How many threads to use when invoking the lossless codec.
    threads: usize,
    _marker: PhantomData<Q>,
}

impl<Q> DASCoder<Q>
where
    Q: Quantizer + CodecParams + Clone,
{
    /// Create a new packer that will always configure the
    /// lossless backend to use exactly `threads` threads.
    pub fn with_threads(threads: usize) -> Self {
        DASCoder {
            threads,
            _marker: PhantomData,
        }
    }

    /// Encode your `f64` array into a single self-describing `Vec<u8>`.
    ///
    /// Header format:
    /// ```text
    /// [magic:"DASP"][ver:1]
    /// [params_len: u32][params…]
    /// [quant_len:  u32][quant_params…]
    /// [height:     u32][width: u32]
    /// [body_len:   u32][payload…]
    /// ```
    pub fn encode(
        &self,
        data: ArrayView2<Q::SourceType>,
        quantizer: &Q,
        params: &CompressParams,
    ) -> Result<Vec<u8>, CodecError> {
        // 1) Serialize params + quantizer
        let params_bytes = params.serialize()?;
        let quant_bytes = quantizer.serialize()?;

        // 2) Build a lossless codec with the right thread count,
        //    wrap it in our lossy adapter, and compress.
        let lossless = LosslessCodec::with_threads(params.clone(), self.threads)?;
        let lossy = LossyCodec::new(lossless, quantizer.clone());
        let body = lossy.compress(data)?;

        // 3) Write header + body
        let (h, w) = data.dim();
        let mut out = Vec::new();
        out.extend(b"DASP");                  // magic
        out.push(1);                          // version
        out.extend(&(params_bytes.len() as u32).to_le_bytes());
        out.extend(&params_bytes);
        out.extend(&(quant_bytes.len()  as u32).to_le_bytes());
        out.extend(&quant_bytes);
        out.extend(&(h as u32).to_le_bytes());
        out.extend(&(w as u32).to_le_bytes());
        out.extend(&(body.len() as u32).to_le_bytes());
        out.extend(&body);
        Ok(out)
    }

    /// Decode a DASPack bitstream back into an `Array2<f64>`.
    ///
    /// All header fields (params, quantizer, shape) are read
    /// from the stream; we re-create the `LosslessCodec` with
    /// the same thread count as on the encoder side.
    pub fn decode(&self, stream: &[u8]) -> Result<Array2<Q::SourceType>> {
        let mut off = 0;

        debug_assert_eq!(&stream[off..off + 4], b"DASP", "invalid magic");
        off += 4;
        let ver = stream[off]; off += 1;
        debug_assert_eq!(ver, 1, "unsupported version");

        // read params
        let p_len = u32::from_le_bytes(stream[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let p_bytes = &stream[off..off + p_len];
        off += p_len;
        let params = CompressParams::read(p_bytes);

        // read quantizer params
        let q_len = u32::from_le_bytes(stream[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let q_bytes = &stream[off..off + q_len];
        off += q_len;
        let quantizer: Q = Q::read(q_bytes);

        // read full shape
        let h = u32::from_le_bytes(stream[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let w = u32::from_le_bytes(stream[off..off + 4].try_into().unwrap()) as usize;
        off += 4;

        // read payload
        let b_len = u32::from_le_bytes(stream[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let body = &stream[off..off + b_len];

        // rebuild and run the codec
        let lossless = LosslessCodec::with_threads(params.clone(), self.threads)?;
        let lossy = LossyCodec::new(lossless, quantizer);
        lossy.decompress(body, (h, w))
    }
}


// Add these tests to the bottom of your `daspacker.rs`

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::UniformQuantizer;
    use ndarray::{arr2, Array2};
    use rand::{Rng, SeedableRng};
    use rand_distr::Uniform;

    /// Helper: quantize → dequantize to get expected float output
    fn expect_quantized(data: &Array2<f64>, step: f64) -> Array2<f64> {
        let quant = UniformQuantizer::new(step as f32);
        let ints = quant.quantize(data.view());
        quant.dequantize(ints.view())
    }

    /// Core roundtrip verifier
    fn verify_roundtrip(
        threads: usize,
        step: f64,
        shape: (usize, usize),
        block: (usize, usize),
    ) {
        // Setup packer
        let packer = DASCoder::<UniformQuantizer>::with_threads(threads);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let uniform = Uniform::new(-100.0, 100.0).unwrap();

        // Random data
        let data: Array2<f64> = Array2::from_shape_fn(shape, |_| rng.sample(uniform));
        let quantizer = UniformQuantizer::new(step as f32);
        let params = CompressParams::new(block.0, block.1, /*lx*/ 1, /*lt*/ 1, /*order*/ 2);

        // Encode → Decode
        let bytes = packer.encode(data.view(), &quantizer, &params)
            .expect("encode failed");
        // Verify header is non-empty
        assert!(bytes.len() > 20, "stream too small");

        let out = packer.decode(&bytes).expect("decode failed");
        assert_eq!(out.dim(), shape);

        // Compare to expected quantized values
        let expected = expect_quantized(&data, step);
        let tol = step / 2.0 + 1e-12;
        for ((i, j), &exp) in expected.indexed_iter() {
            let got = out[[i, j]];
            assert!((got - exp).abs() <= tol,
                "mismatch at ({},{}): got {} vs {}", i, j, got, exp
            );
        }
    }

    #[test]
    fn roundtrip_small_one_thread() {
        verify_roundtrip(1, 0.5, (3, 5), (2, 3));
    }

    #[test]
    fn roundtrip_small_multi_thread() {
        verify_roundtrip(4, 0.5, (3, 5), (2, 3));
    }

    #[test]
    fn roundtrip_large_single_block() {
        verify_roundtrip(2, 0.1, (64, 64), (64, 64));
    }

    #[test]
    fn roundtrip_large_multi_block() {
        verify_roundtrip(8, 0.05, (128, 128), (16, 16));
    }


    #[test]
    fn different_params_change_stream() {
        let packer = DASCoder::<UniformQuantizer>::with_threads(1);
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let q1 = UniformQuantizer::new(0.1);
        let p1 = CompressParams::new(2, 2, 1, 1, 1);
        let q2 = UniformQuantizer::new(0.2);
        let p2 = CompressParams::new(2, 2, 1, 1, 1);

        let b1 = packer.encode(data.view(), &q1, &p1).unwrap();
        let b2 = packer.encode(data.view(), &q2, &p2).unwrap();
        assert_ne!(b1, b2, "streams should differ when quant params change");
    }
}


#[cfg(test)]
mod lossless_roundtrip_tests {
    use super::*;
    use crate::codec::LosslessQuantizer;
    use ndarray::{Array2};
    use rand::{Rng, SeedableRng};
    use rand_distr::Uniform;

    /// Core lossless roundtrip verifier for integer data
    fn verify_lossless_roundtrip(
        threads: usize,
        shape: (usize, usize),
        range: std::ops::Range<i32>,
    ) {
        // Setup packer
        let packer = DASCoder::<LosslessQuantizer>::with_threads(threads);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEADBEEF);
        let dist = Uniform::new(range.start, range.end).unwrap();

        // Random integer data
        let data: Array2<i32> = Array2::from_shape_fn(shape, |_| rng.sample(dist));
        let quantizer = LosslessQuantizer;
        let params = CompressParams::new(shape.0, shape.1, /*lx*/ 0, /*lt*/ 0, /*order*/ 0);

        // Encode → Decode
        let bytes = packer.encode(data.view(), &quantizer, &params)
            .expect("lossless encode failed");
        // Ensure header is present
        assert!(bytes.len() > 4, "stream too small");

        let out = packer.decode(&bytes).expect("lossless decode failed");
        assert_eq!(out.dim(), shape);

        // Compare exactly
        for ((i, j), &orig) in data.indexed_iter() {
            let got = out[[i, j]];
            assert_eq!(got, orig, "mismatch at ({},{}): got {} vs {}", i, j, got, orig);
        }
    }

    #[test]
    fn lossless_roundtrip_small_one_thread() {
        verify_lossless_roundtrip(1, (4, 4), -50..50);
    }

    #[test]
    fn lossless_roundtrip_small_multi_thread() {
        verify_lossless_roundtrip(4, (4, 4), -50..50);
    }

    #[test]
    fn lossless_roundtrip_large_single_thread() {
        verify_lossless_roundtrip(1, (64, 128), -1000..1000);
    }

    #[test]
    fn lossless_roundtrip_large_multi_thread() {
        verify_lossless_roundtrip(8, (128, 64), -1000..1000);
    }
}
