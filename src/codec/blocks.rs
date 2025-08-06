//! Block-wise lossless compression with LPC prediction, row-wise mean removal,
//! and wavelet sub-band entropy coding.

use std::io::{Cursor, Read};

use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::{s, Array2, ArrayView2, Axis};
use crate::core::entropy::{compress_residuals_rice, decompress_residuals_rice};
use crate::core::prediction::{MultiBlockPredictor};

use super::params::{CompressParams, Shape};

// ───────────────────────── INTERNAL HELPERS ────────────────────────────

#[inline]
fn split_subbands_2d(
    res_block: &Array2<i32>,
    lx: usize,
    lt: usize,
) -> Vec<ArrayView2<i32>> {
    if lx > 0 && lt > 0 {
        let (h, w) = res_block.dim();
        let (h2, w2) = (h / 2, w / 2);
        vec![
            res_block.slice(s![0..h2, 0..w2]),
            res_block.slice(s![0..h2, w2..]),
            res_block.slice(s![h2.., 0..w2]),
            res_block.slice(s![h2.., w2..]),
        ]
    } else {
        vec![res_block.view()]
    }
}


#[inline]
fn combine_subbands_2d(subbands: &[Array2<i32>], lx: usize, lt: usize) -> Array2<i32> {
    if lx > 0 && lt > 0 {
        let h_top = subbands[0].nrows();
        let h_bot = subbands[2].nrows();
        let w_left = subbands[0].ncols();
        let w_right = subbands[1].ncols();

        let mut out = Array2::<i32>::zeros((h_top + h_bot, w_left + w_right));
        out.slice_mut(s![0..h_top, 0..w_left]).assign(&subbands[0]);
        out.slice_mut(s![0..h_top, w_left..]).assign(&subbands[1]);
        out.slice_mut(s![h_top.., 0..w_left]).assign(&subbands[2]);
        out.slice_mut(s![h_top.., w_left..]).assign(&subbands[3]);
        out
    } else {
        subbands[0].clone()
    }
}

// ────────────────────────── MEAN (DE)COMPRESSION ───────────────────────

#[inline]
fn compress_means(means: &[i32]) -> Result<Vec<u8>> {
    if means.is_empty() {
        return Ok(Vec::new());
    }
    let mut deltas = Vec::<i32>::with_capacity(means.len());
    deltas.push(means[0]);
    for i in 1..means.len() {
        deltas.push(means[i] - means[i - 1]);
    }
    let arr = Array2::from_shape_vec((deltas.len(), 1), deltas)?;
    Ok(compress_residuals_rice(arr.view())?)
}

#[inline]
fn decompress_means(stream: &[u8], rows: usize) -> Result<Vec<i32>> {
    if stream.is_empty() {
        return Ok(vec![0; rows]);
    }
    let deltas = decompress_residuals_rice(stream)?;
    let mut means = Vec::with_capacity(rows);
    for (i, &d) in deltas.iter().enumerate() {
        let m = if i == 0 { d } else { means[i - 1] + d };
        means.push(m);
    }
    Ok(means)
}

// ──────────────────────────── BLOCK TYPES ──────────────────────────────

/// Fully-owned encoded block.
#[derive(Default, Clone)]
pub struct EncodedBlock {
    residuals: Vec<u8>,
    means: Vec<u8>,
    row_coefs: Vec<u8>,
    col_coefs: Vec<u8>,
}

impl EncodedBlock {
    /// Serialise `EncodedBlock` into a contiguous byte vector:
    /// [4×u32 section lengths][sections concatenated in fixed order]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.total_len());
        for section in [&self.residuals, &self.means, &self.row_coefs, &self.col_coefs] {
            buf.write_u32::<LittleEndian>(section.len() as u32)
                .unwrap();
        }
        buf.extend_from_slice(&self.residuals);
        buf.extend_from_slice(&self.means);
        buf.extend_from_slice(&self.row_coefs);
        buf.extend_from_slice(&self.col_coefs);
        buf
    }

    #[inline]
    fn total_len(&self) -> usize {
        16 + self.residuals.len() + self.means.len() + self.row_coefs.len() + self.col_coefs.len()
    }
}

/// Lightweight view into an encoded block backed by a single byte slice.
pub struct EncodedBlockSlice<'a> {
    pub residuals: &'a [u8],
    pub means: &'a [u8],
    pub row_coefs: &'a [u8],
    pub col_coefs: &'a [u8],
}

impl<'a> EncodedBlockSlice<'a> {
    /// Construct from the contiguous buffer produced by `EncodedBlock::to_bytes`.
    pub fn from_bytes(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < 16 {
            return Err(anyhow!("Encoded block too small"));
        }
        let mut cur = Cursor::new(buf);
        let r_len = cur.read_u32::<LittleEndian>()? as usize;
        let m_len = cur.read_u32::<LittleEndian>()? as usize;
        let row_len = cur.read_u32::<LittleEndian>()? as usize;
        let col_len = cur.read_u32::<LittleEndian>()? as usize;

        let mut offset = 16;
        let residuals = &buf[offset..offset + r_len];
        offset += r_len;
        let means = &buf[offset..offset + m_len];
        offset += m_len;
        let row_coefs = &buf[offset..offset + row_len];
        offset += row_len;
        let col_coefs = &buf[offset..offset + col_len];

        Ok(Self {
            residuals,
            means,
            row_coefs,
            col_coefs,
        })
    }
}

// ─────────────────────────── ROW-MEAN UTILITIES ────────────────────────

fn remove_row_means(block: &mut Array2<i32>) -> Vec<i32> {
    let (rows, cols) = block.dim();
    let mut means = Vec::with_capacity(rows);
    for mut row in block.axis_iter_mut(Axis(0)) {
        let sum: i64 = row.iter().map(|&v| v as i64).sum();
        let mean = (sum / cols as i64) as i32;
        means.push(mean >> 4); // quantise
        for x in row.iter_mut() {
            *x -= (mean >> 4) << 4;
        }
    }
    means
}

fn add_row_means(block: &mut Array2<i32>, means: &[i32]) {
    for (mut row, &m) in block.axis_iter_mut(Axis(0)).zip(means) {
        for x in row.iter_mut() {
            *x += m << 4; // dequantize mean
        }
    }
}

// ──────────────────────────── BLOCK TRANSFORM ──────────────────────────

/// A generic codec over rectangular blocks.
pub trait BlockTransform: Send + Sync {
    fn encode(&self, blk: ArrayView2<i32>) -> Result<EncodedBlock>;
    fn decode(&self, blk: EncodedBlockSlice, shape: Shape) -> Result<Array2<i32>>;
}

// ──────────────────────────── BLOCK PROCESSOR ──────────────────────────

pub struct BlockProcessor {
    p: CompressParams,
}

impl BlockProcessor {
    pub fn new(p: CompressParams) -> Self {
        Self { p }
    }
}

impl BlockTransform for BlockProcessor {
    fn encode(&self, blk: ArrayView2<i32>) -> Result<EncodedBlock> {
        // Work on a private copy only when needed.
        let mut block = blk.to_owned();

        // 1. Optional DC removal (row means)
        let means = if self.p.row_demean {
            remove_row_means(&mut block)
        } else {
            Vec::new()
        };

        // 2. LPC prediction
        let predictor =
            MultiBlockPredictor::new(self.p.lx, self.p.lt, self.p.lpc_order, self.p.lpc_bits, self.p.lpc_range);
        let (residuals, row_c, col_c) = predictor.predict_diff(&block);

        // 3. Compress residual sub-bands 
        let residual_bytes = {
            let mut out = Vec::with_capacity(blk.len() >> 2);
            for sb in split_subbands_2d(&residuals, self.p.lx, self.p.lt) {
                let comp = compress_residuals_rice(sb)?;
                out.write_u32::<LittleEndian>(comp.len() as u32)?;
                out.extend_from_slice(&comp);
            }
            out
        };

        // 4. Compress means 
        let mean_bytes = if self.p.row_demean {
            compress_means(&means)?
        } else {
            Vec::new()
        };

        // 5. Quantise LPC coeffs
        let row_bytes: Vec<u8> = row_c
            .iter()
            .map(|&c| predictor.lpc_tool.quantize_uniform(c) as u8)
            .collect();
        let col_bytes: Vec<u8> = col_c
            .iter()
            .map(|&c| predictor.lpc_tool.quantize_uniform(c) as u8)
            .collect();

        Ok(EncodedBlock {
            residuals: residual_bytes,
            means: mean_bytes,
            row_coefs: row_bytes,
            col_coefs: col_bytes,
        })
    }

    fn decode(&self, sl: EncodedBlockSlice, shape: Shape) -> Result<Array2<i32>> {
        let (rows, cols) = shape;

        // Residuals
        let residuals = {
            let mut cur = Cursor::new(sl.residuals);
            let sb_cnt = if self.p.lx > 0 && self.p.lt > 0 { 4 } else { 1 };
            let mut subbands = Vec::with_capacity(sb_cnt);
            for _ in 0..sb_cnt {
                let len = cur.read_u32::<LittleEndian>()? as usize;
                let mut buf = vec![0u8; len];
                cur.read_exact(&mut buf)?;
                subbands.push(decompress_residuals_rice(&buf)?);
            }
            combine_subbands_2d(&subbands, self.p.lx, self.p.lt)
        };

        anyhow::ensure!(
            residuals.len() == rows * cols,
            "residuals carry {} samples, expected {}×{} = {}",
            residuals.len(), rows, cols, rows * cols
        );

        // Means
        let means = if self.p.row_demean {
            decompress_means(sl.means, rows)?
        } else {
            vec![0; rows]
        };

        // LPC coeffs
        let predictor =
            MultiBlockPredictor::new(self.p.lx, self.p.lt, self.p.lpc_order, self.p.lpc_bits, self.p.lpc_range);

        let k = self.p.lpc_order + 1;

        let row_coefs = {
            let symbols: Vec<u32> = sl.row_coefs.iter().map(|&b| b as u32).collect();
            let vals = predictor.lpc_tool.coefs_from_symbols(&symbols);
            anyhow::ensure!(vals.len() == rows * k,
                "row_coefs decoded length {} != rows({}) * (order+1)({})", vals.len(), rows, k);
            Array2::from_shape_vec((rows, k), vals)?
        };
        let col_coefs = {
            let symbols: Vec<u32> = sl.col_coefs.iter().map(|&b| b as u32).collect();
            let vals = predictor.lpc_tool.coefs_from_symbols(&symbols);
            anyhow::ensure!(vals.len() == cols * k,
                "col_coefs decoded length {} != cols({}) * (order+1)({})", vals.len(), cols, k);
            Array2::from_shape_vec((cols, k), vals)?
        };

        // Reconstruct (no extra padding anymore)
        let mut block = predictor.reconstruct_diff(residuals, &row_coefs, &col_coefs);

        if self.p.row_demean {
            add_row_means(&mut block, &means);
        }
        Ok(block)
    }
}

/* ────────────────────────────────────────────────────────────────────
 * Unit-tests
 * ────────────────────────────────────────────────────────────────── */

 #[cfg(test)]
mod means_codec_tests {
    use super::*;

    /// Helper: round-trip `qs` (quantised means) through the codec and
    /// make sure we get the *de-quantised* values (`qs << 4`) back.
    fn roundtrip(qs: &[i32]) {
        // encode
        let bytes = compress_means(qs).expect("compress_means failed");

        // decode – rows == input-len
        let recon = decompress_means(&bytes, qs.len())
            .expect("decompress_means failed");

        // expected de-quantised means
        let expected = qs;
        assert_eq!(recon, expected);
    }

    /// 1. Empty input
    #[test]
    fn empty_input() {
        let qs: Vec<i32> = Vec::new();
        let bytes = compress_means(&qs).expect("compress failed");
        assert!(bytes.is_empty());

        let recon = decompress_means(&bytes, /*rows=*/0).expect("decompress failed");
        assert!(recon.is_empty());
    }

    /// 2. Single mean value
    #[test]
    fn single_value() {
        let qs = vec![3];          // quantised   (really 48 after <<4)
        roundtrip(&qs);
    }

    /// 3. Sequence with varying deltas (+, – , large, small)
    #[test]
    fn varying_deltas() {
        // Quantised values:     4,  7, -1, -1, 12, -8
        // Corresponding means: 64,112,-16,-16,192,-128
        let qs = vec![4, 7, -1, -1, 12, -8];
        roundtrip(&qs);
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    /// Helper that builds a ready-to-use BlockProcessor.
    fn make_codec(p: CompressParams) -> BlockProcessor {
        BlockProcessor::new(p)
    }

    /// Round-trip a single block and assert bit-exact equality.
    fn roundtrip(codec: &BlockProcessor, blk: &Array2<i32>) {
        // Encode
        let enc = codec.encode(blk.view()).expect("encode failed");

        // Serialise and create slice-view
        let buf = enc.to_bytes();
        let slice = EncodedBlockSlice::from_bytes(&buf).expect("slice parse");

        // Decode
        let out = codec
            .decode(slice, blk.dim())
            .expect("decode failed");

        assert_eq!(blk, &out);
    }

    #[test]
    fn small_fixed_block() {
        let blk = array![
            [12, -34, 56, -78],
            [90, -12, 34, -56],
            [78, -90, 12, -34]
        ];
        let p = CompressParams::new(3, 4, /*lx=*/1, /*lt=*/1, /*order=*/2);
        let codec = make_codec(p);
        roundtrip(&codec, &blk);
    }

    #[test]
    fn small_fixed_block2() {
        let blk = array![
            [12, -34, 56, -78],
            [90, -12, 34, -56],
            [78, -90, 12, -34]
        ];
        let p = CompressParams::new(3, 4, /*lx=*/0, /*lt=*/0, /*order=*/1);
        let codec = make_codec(p);
        roundtrip(&codec, &blk);
    }

    #[test]
    fn small_fixed_block3() {
        let blk = array![
            [12, -34, 56, -78],
            [90, -12, 34, -56],
            [78, -90, 12, -34]
        ];
        let p = CompressParams::new(3, 4, /*lx=*/0, /*lt=*/0, /*order=*/0);
        let codec = make_codec(p);
        roundtrip(&codec, &blk);
    }

    #[test]
    fn random_uniform_block() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let blk: Array2<i32> =
            Array2::from_shape_fn((64, 32), |_| rng.random_range(-32_768..32_768));
        let p = CompressParams::new(64, 32, 0, 0, 1); // no sub-bands, first-order LPC
        let codec = make_codec(p);
        roundtrip(&codec, &blk);
    }

    #[test]
    fn random_gaussian_block_with_subbands() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEADBEEF);
        let normal = Normal::new(0.0, 10_000.0).unwrap();
        let blk: Array2<i32> = Array2::from_shape_fn((128, 128), |_| {
            normal.sample(&mut rng) as i32
        });

        let mut p = CompressParams::new(128, 128, 1, 1, 4); // 4-tap LPC, 2×2 sub-bands
        p.row_demean = true;
        let codec = make_codec(p);
        roundtrip(&codec, &blk);
    }

    #[test]
    fn random_gaussian_block_with_subbands_nonsquare() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 10_000.0).unwrap();
        let blk: Array2<i32> = Array2::from_shape_fn((128, 128), |_| {
            normal.sample(&mut rng) as i32
        });

        let mut p = CompressParams::new(128, 64, 1, 1, 4); // 4-tap LPC, 2×2 sub-bands
        p.row_demean = true;
        let codec = make_codec(p);
        roundtrip(&codec, &blk);
    }

    #[test]
    fn random_gaussian_block_with_subbands_nonsquare2() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(43);
        let normal = Normal::new(0.0, 10_000.0).unwrap();
        let blk: Array2<i32> = Array2::from_shape_fn((128, 128), |_| {
            normal.sample(&mut rng) as i32
        });

        let mut p = CompressParams::new(128, 48, 1, 1, 4); // 4-tap LPC, 2×2 sub-bands
        p.row_demean = true;
        let codec = make_codec(p);
        roundtrip(&codec, &blk);
    }

    #[test]
    fn random_gaussian_block_without_subbands() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEADBEEF);
        let normal = Normal::new(0.0, 10_000.0).unwrap();
        let blk: Array2<i32> = Array2::from_shape_fn((128, 128), |_| {
            normal.sample(&mut rng) as i32
        });

        let mut p = CompressParams::new(128, 128, 0, 0, 4); // 4-tap LPC, 2×2 sub-bands
        p.row_demean = true;
        let codec = make_codec(p);
        roundtrip(&codec, &blk);
    }
}

