// lossless.rs ──────────────────────────────────────────────────
// Multithreaded, loss-less block codec built on BlockProcessor.
// --------------------------------------------------------------------
use std::io::{Cursor, Read};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

use super::blocks::{BlockProcessor, BlockTransform, EncodedBlock, EncodedBlockSlice};
use super::params::{CompressParams, Shape};
use super::common;
use super::Codec;

// ─────────────────────── LOSSLESS IMPLEMENTATION ─────────────────────
pub struct LosslessCodec {
    p: CompressParams,
    inner: Arc<BlockProcessor>,
}

impl LosslessCodec {
    /// Build a new codec.
    /// Semantics:
    ///   - `threads > 0`  → use exactly that many threads (global Rayon pool).
    ///   - `threads == 0` → use DASPACK_THREADS env (positive integer), else 1 thread.
    pub fn with_threads(p: CompressParams, threads: usize) -> anyhow::Result<Self> {
        let n = common::effective_threads(threads);

        // Initialize (or attempt to) the Rayon *global* pool once, using `n`.
        // If the global pool was already initialized elsewhere, we log and continue.
        common::ensure_global_rayon_pool(n);

        Ok(Self {
            inner: std::sync::Arc::new(BlockProcessor::new(p.clone())),
            p,
        })
    }

    pub fn new(p: CompressParams) -> anyhow::Result<Self> { Self::with_threads(p, 0) }
}

// ────────────────────────── CODEC TRAIT IMPL ─────────────────────────
impl Codec for LosslessCodec {
    fn compress(&self, data: ArrayView2<i32>) -> Result<Vec<u8>> {
        let coords = block_coords(data.dim(), &self.p);
        let blocks: Vec<Vec<u8>> = coords
            .par_iter()
            .map(|&(r, c)| {
                let view = slice_block(data, (r, c), &self.p);
                let enc: EncodedBlock = self.inner.encode(view)?;
                let mut buf = enc.to_bytes();

                let mut out = Vec::with_capacity(4 + buf.len());
                out.write_u32::<LittleEndian>(buf.len() as u32)?;
                out.append(&mut buf);
                Ok::<_, anyhow::Error>(out)
            })
            .collect::<Result<_>>()?;

        let mut stream = Vec::new();
        stream.write_u32::<LittleEndian>(blocks.len() as u32)?;
        for b in blocks {
            stream.extend_from_slice(&b);
        }
        Ok(stream)
    }

    fn decompress(&self, stream: &[u8], shape: Shape) -> Result<Array2<i32>> {
        let (h, w) = shape;
        let coords = block_coords(shape, &self.p);

        let true_shapes: Vec<(usize, usize)> = coords
            .iter()
            .map(|&(r, c)| {
                let rows = (r + self.p.block_height).min(h) - r;
                let cols = (c + self.p.block_width).min(w) - c;
                (rows, cols)
            })
            .collect();

        let mut cur = Cursor::new(stream);
        let n_blocks = cur.read_u32::<LittleEndian>()? as usize;
        if n_blocks != coords.len() {
            return Err(anyhow!(
                "Block-count mismatch (stream {n_blocks}, expected {})",
                coords.len()
            ));
        }

        let mut raw = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let len = cur.read_u32::<LittleEndian>()? as usize;
            let mut buf = vec![0u8; len];
            cur.read_exact(&mut buf)?;
            raw.push(buf);
        }

        let decoded: Vec<(usize, Array2<i32>)> = raw
            .into_par_iter()
            .enumerate()
            .map(|(idx, buf)| {
                let slice = EncodedBlockSlice::from_bytes(&buf)?;
                let blk_shape = true_shapes[idx];
                let blk = self.inner.decode(slice, blk_shape)?;
                Ok::<_, anyhow::Error>((idx, blk))
            })
            .collect::<Result<_>>()?;

        let mut out = Array2::<i32>::zeros((h, w));
        for (idx, blk) in decoded {
            let (r0, c0) = coords[idx];
            out.slice_mut(s![r0..r0 + blk.nrows(), c0..c0 + blk.ncols()])
                .assign(&blk);
        }
        Ok(out)
    }
}

// ────────────────────────── HELPER FUNCTIONS ─────────────────────────
fn block_coords((h, w): Shape, p: &CompressParams) -> Vec<(usize, usize)> {
    (0..h)
        .step_by(p.block_height)
        .flat_map(|r| (0..w).step_by(p.block_width).map(move |c| (r, c)))
        .collect()
}

fn slice_block<'a>(
    data: ArrayView2<'a, i32>,
    (r, c): (usize, usize),
    p: &CompressParams,
) -> ArrayView2<'a, i32> {
    let r_end = (r + p.block_height).min(data.nrows());
    let c_end = (c + p.block_width).min(data.ncols());
    data.slice_move(s![r..r_end, c..c_end])
}

#[cfg(test)]
mod codec_tests {
    use super::*;               // brings Codec, CodecLossless, CompressParams into scope
    use ndarray::{array, Array2};
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    /// Utility: compress → decompress → assert equality.
    fn roundtrip<C: Codec>(codec: &C, data: &Array2<i32>) {
        let bytes = codec.compress(data.view()).expect("compress");
        let out = codec
            .decompress(&bytes, data.dim())
            .expect("decompress");
        assert_eq!(data, &out, "round-trip mismatch");
    }

    #[test]
    fn fixed_small_block() {
        let data = array![
            [12, -34, 56, -78],
            [90, -12, 34, -56],
            [78, -90, 12, -34]
        ];
        let p = CompressParams::new(3, 4, /*lx*/ 1, /*lt*/ 1, /*order*/ 2);
        let codec = LosslessCodec::with_threads(p, /*threads*/ 4).unwrap();
        roundtrip(&codec, &data);
    }

    #[test]
    fn random_uniform_frame() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let data: Array2<i32> =
            Array2::from_shape_fn((64, 32), |_| rng.random_range(-32_768..32_768));
        let p = CompressParams::new(64, 32, 0, 0, 1); // single block, 1-tap LPC
        let codec = LosslessCodec::new(p).unwrap(); // 0 ⇒ global Rayon pool
        roundtrip(&codec, &data);
    }

    #[test]
    fn random_gaussian_frame() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEADBEEF);
        let normal = Normal::new(0.0, 10_000.0).unwrap();
        let data: Array2<i32> =
            Array2::from_shape_fn((128, 128), |_| normal.sample(&mut rng) as i32);

        let mut p = CompressParams::new(64, 64, /*lx*/ 1, /*lt*/ 1, /*order*/ 4);
        p.row_demean = true;                           // enable DC removal
        let codec = LosslessCodec::with_threads(p, 8).unwrap(); // 8 threads
        roundtrip(&codec, &data);
    }

    #[test]
    fn random_gaussian_frame_nonsquare() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEADBEEF);
        let normal = Normal::new(0.0, 10_000.0).unwrap();
        let data: Array2<i32> =
            Array2::from_shape_fn((128, 128), |_| normal.sample(&mut rng) as i32);

        let mut p = CompressParams::new(64, 48, /*lx*/ 1, /*lt*/ 1, /*order*/ 4);
        p.row_demean = true;                           // enable DC removal
        let codec = LosslessCodec::with_threads(p, 8).unwrap(); // 8 threads
        roundtrip(&codec, &data);
    }
}
