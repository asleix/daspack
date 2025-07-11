//! Block‑wise lossless compression with LPC prediction, row‑wise mean removal,
//! and wavelet sub‑band entropy coding.
//!
//! Public API
//! ----------
//! * [`compress_lossless`] – encode `Array2<i32>` into a single `Vec<u8>`.
//! * [`decompress_lossless`] – round‑trip the bit‑stream back to the original
//!   2‑D data.
//!

use std::cmp;
use std::io::{Cursor, Read};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use numpy::ndarray::{concatenate, s, Array2, Axis};
use rayon::prelude::*;
use crate::entropy::{compress_residuals, decompress_residuals};
use crate::prediction::{LpcTool, MultiBlockPredictor};

//────────────────────────────── PARAMETERS ──────────────────────────────

/// Encoding parameters (kept outside the bit‑stream).
#[derive(Debug, Clone)]
pub struct CompressParams {
    pub block_height: usize,
    pub block_width: usize,
    pub lx: usize,
    pub lt: usize,
    pub lpc_order: usize,
    pub tail_cut: f64,
    // optional tuning
    pub lpc_bits: u8,
    pub lpc_range: (f64, f64),

    pub row_demean: bool,
}

impl CompressParams {
    pub fn new(block_height: usize, block_width: usize, lx: usize, lt: usize, lpc_order: usize, tail_cut: f64) -> Self {
        Self { block_height, block_width, lx, lt, lpc_order, tail_cut, lpc_bits: 6, lpc_range: (-1.5, 1.5), row_demean: true}
    }
}

//─────────────────────────── INTERNAL HELPERS ───────────────────────────

#[inline]
fn split_subbands_2d(res_block: &Array2<i32>, lx: usize, lt: usize) -> Vec<Array2<i32>> {
    if lx > 0 && lt > 0 {
        let (h, w) = res_block.dim();
        let (h2, w2) = (h / 2, w / 2);
        vec![
            res_block.slice(s![0..h2, 0..w2]).to_owned(),
            res_block.slice(s![0..h2, w2..]).to_owned(),
            res_block.slice(s![h2.., 0..w2]).to_owned(),
            res_block.slice(s![h2.., w2..]).to_owned(),
        ]
    } else {
        vec![res_block.to_owned()]
    }
}

#[inline]
fn combine_subbands_2d(subbands: &[Array2<i32>], lx: usize, lt: usize) -> Array2<i32> {
    if lx > 0 && lt > 0 {
        let (h2, w2) = (subbands[0].nrows(), subbands[0].ncols());
        let mut out = Array2::<i32>::zeros((h2 * 2, w2 * 2));
        out.slice_mut(s![0..h2, 0..w2]).assign(&subbands[0]);
        out.slice_mut(s![0..h2, w2..]).assign(&subbands[1]);
        out.slice_mut(s![h2.., 0..w2]).assign(&subbands[2]);
        out.slice_mut(s![h2.., w2..]).assign(&subbands[3]);
        out
    } else {
        subbands[0].clone()
    }
}

//────────────────────────── MEAN (DE)COMPRESSION ────────────────────────

#[inline]
fn compress_means(means: &[i32], _tail_cut: f64) -> Result<Vec<u8>> {
    if means.is_empty() { return Ok(Vec::new()); }
    let mut deltas = Vec::<i32>::with_capacity(means.len());
    deltas.push(means[0]);
    for i in 1..means.len() { deltas.push(means[i] - means[i - 1]); }
    let arr = Array2::from_shape_vec((deltas.len(), 1), deltas)?;
    Ok(compress_residuals(&arr)?)
}

#[inline]
fn decompress_means(stream: &[u8], rows: usize) -> Result<Vec<i32>> {
    if stream.is_empty() { return Ok(vec![0; rows]); }
    let deltas = decompress_residuals(stream)?;
    let mut means = Vec::with_capacity(rows);
    for (i, &d) in deltas.iter().enumerate() {
        let m = if i == 0 { d } else { means[i - 1] + d };
        means.push(m << 4); // dequantize
    }
    Ok(means)
}

//───────────────────────────── BLOCK TYPES ──────────────────────────────

#[derive(Default, Clone)]
struct EncodedBlock {
    residuals: Vec<u8>,
    means: Vec<u8>,
    row_coefs: Vec<u8>,
    col_coefs: Vec<u8>,
}
impl EncodedBlock {
    #[inline] fn len(&self) -> usize { 16 + self.residuals.len() + self.means.len() + self.row_coefs.len() + self.col_coefs.len() }
}

struct EncodedBlockSlice<'a> {
    residuals: &'a [u8],
    means: &'a [u8],
    row_coefs: &'a [u8],
    col_coefs: &'a [u8],
}

//────────────────────────── ROW‑MEAN UTILITIES ──────────────────────────

fn remove_row_means(block: &mut Array2<i32>) -> Vec<i32> {
    let (rows, cols) = block.dim();
    let mut means = Vec::with_capacity(rows);
    for mut row in block.axis_iter_mut(Axis(0)) {
        let sum: i64 = row.iter().map(|&v| v as i64).sum();
        let mean = (sum / cols as i64) as i32;
        means.push(mean >> 4); // quantize mean
        for x in row.iter_mut() { *x -= (mean >> 4) << 4; }
    }
    means
}

fn add_row_means(block: &mut Array2<i32>, means: &[i32]) {
    for (mut row, &m) in block.axis_iter_mut(Axis(0)).zip(means) {
        for x in row.iter_mut() { *x += m; }
    }
}

//──────────────────────────── BLOCK PROCESSOR ───────────────────────────

struct BlockProcessor {
    p: CompressParams,
    lpc_tool: LpcTool,
}

impl BlockProcessor {
    fn new(p: CompressParams, lpc_tool: LpcTool) -> Self { Self { p, lpc_tool } }

    fn encode_block(&self, mut block: Array2<i32>) -> Result<EncodedBlock> {
        // 1. Optional DC removal (row means)
        let means = if self.p.row_demean {
            remove_row_means(&mut block)
        } else {
            Vec::new()
        };

        // 2. LPC prediction
        let predictor = MultiBlockPredictor::new(self.p.lx, self.p.lt, self.lpc_tool.clone());
        let (residuals, mut row_c, mut col_c) = predictor.predict_diff(&block);
        if self.p.lpc_order > 0 {
            row_c = row_c.slice(s![.., 1..]).to_owned();
            col_c = col_c.slice(s![.., 1..]).to_owned();
        }

        // 3. Compress residual sub‑bands
        let residual_bytes = {
            let subbands = split_subbands_2d(&residuals, self.p.lx, self.p.lt);
            let mut out = Vec::with_capacity(subbands.len() * 4);
            for sb in subbands {
                let comp = compress_residuals(&sb)?;
                out.write_u32::<LittleEndian>(comp.len() as u32)?;
                out.extend_from_slice(&comp);
            }
            out
        };

        // 4. Compress means (may be empty)
        let mean_bytes = if self.p.row_demean {
            compress_means(&means, self.p.tail_cut)?
        } else {
            Vec::new()
        };

        // 5. Quantise LPC coeffs
        let row_bytes: Vec<u8> = row_c.iter().map(|&c| self.lpc_tool.quantize_uniform(c) as u8).collect();
        let col_bytes: Vec<u8> = col_c.iter().map(|&c| self.lpc_tool.quantize_uniform(c) as u8).collect();

        Ok(EncodedBlock { residuals: residual_bytes, means: mean_bytes, row_coefs: row_bytes, col_coefs: col_bytes })
    }

    fn decode_block(&self, slice: EncodedBlockSlice, rows: usize, cols: usize) -> Result<Array2<i32>> {
        // Residuals
        let residuals = {
            let mut cur = Cursor::new(slice.residuals);
            let sb_cnt = if self.p.lx > 0 && self.p.lt > 0 { 4 } else { 1 };
            let mut subbands = Vec::with_capacity(sb_cnt);
            for _ in 0..sb_cnt {
                let len = cur.read_u32::<LittleEndian>()? as usize;
                let mut buf = vec![0u8; len];
                cur.read_exact(&mut buf)?;
                subbands.push(decompress_residuals(&buf)?);
            }
            combine_subbands_2d(&subbands, self.p.lx, self.p.lt)
        };

        // Means
        // Means (may be absent)
        let means = if self.p.row_demean {
            decompress_means(slice.means, rows)?
        } else {
            vec![0; rows]   // dummy, never added
        };

        // LPC coeffs
        let row_coefs = {
            let symbols: Vec<u32> = slice.row_coefs.iter().map(|&b| b as u32).collect();
            let vals = self.lpc_tool.coefs_from_symbols(&symbols);
            Array2::from_shape_vec((rows, self.p.lpc_order), vals)?
        };
        let col_coefs = {
            let symbols: Vec<u32> = slice.col_coefs.iter().map(|&b| b as u32).collect();
            let vals = self.lpc_tool.coefs_from_symbols(&symbols);
            Array2::from_shape_vec((cols, self.p.lpc_order), vals)?
        };
        let full_row = if self.p.lpc_order > 0 {
            let ones = Array2::from_elem((rows, 1), 1.0);
            concatenate(Axis(1), &[ones.view(), row_coefs.view()])?
        } else { row_coefs };
        let full_col = if self.p.lpc_order > 0 {
            let ones = Array2::from_elem((cols, 1), 1.0);
            concatenate(Axis(1), &[ones.view(), col_coefs.view()])?
        } else { col_coefs };

        // Reconstruct
        let predictor = MultiBlockPredictor::new(self.p.lx, self.p.lt, self.lpc_tool.clone());
        let mut block = predictor.reconstruct_diff(residuals, &full_row, &full_col);
        
        if self.p.row_demean {
            add_row_means(&mut block, &means);
        }

        Ok(block)
    }
}

//──────────────────────────── TOP‑LEVEL API ─────────────────────────────

pub fn compress_lossless(data: &Array2<i32>, p: CompressParams) -> Result<Vec<u8>> {
    let lpc_tool = LpcTool::new(p.lpc_order, p.lpc_bits, p.lpc_range.1, p.lpc_range.0);
    let bp = BlockProcessor::new(p.clone(), lpc_tool);
    let (h, w) = data.dim();
    let blocks: Vec<(usize, usize)> = (0..h)
        .step_by(p.block_height)
        .flat_map(|r| (0..w).step_by(p.block_width).map(move |c| (r, c)))
        .collect();

    let enc_blocks: Vec<EncodedBlock> = blocks.par_iter().map(|&(r, c)| {
        let r_end = cmp::min(r + p.block_height, h);
        let c_end = cmp::min(c + p.block_width, w);
        let blk = data.slice(s![r..r_end, c..c_end]).to_owned();
        bp.encode_block(blk).with_context(|| format!("encode block @({r},{c})"))
    }).collect::<Result<_>>()?;

    let total_len: usize = enc_blocks.iter().map(EncodedBlock::len).sum();
    let mut stream = Vec::with_capacity(total_len);
    for b in enc_blocks {
        stream.write_u32::<LittleEndian>(b.residuals.len() as u32)?;
        stream.extend_from_slice(&b.residuals);
        stream.write_u32::<LittleEndian>(b.means.len() as u32)?;
        stream.extend_from_slice(&b.means);
        stream.write_u32::<LittleEndian>(b.row_coefs.len() as u32)?;
        stream.extend_from_slice(&b.row_coefs);
        stream.write_u32::<LittleEndian>(b.col_coefs.len() as u32)?;
        stream.extend_from_slice(&b.col_coefs);
    }
    Ok(stream)
}

pub fn decompress_lossless(stream: &[u8], shape: (usize, usize), p: CompressParams) -> Result<Array2<i32>> {
    let lpc_tool = LpcTool::new(p.lpc_order, p.lpc_bits, p.lpc_range.1, p.lpc_range.0);
    let bp = BlockProcessor::new(p.clone(), lpc_tool);
    let (h, w) = shape;

    // prepare block positions
    let blocks: Vec<(usize, usize)> = (0..h)
        .step_by(p.block_height)
        .flat_map(|r| (0..w).step_by(p.block_width).map(move |c| (r, c)))
        .collect();

    // helper to read a length-prefixed slice and advance cursor
    let mut cur = Cursor::new(stream);
    let read_prefixed = |cur: &mut Cursor<&[u8]>| -> Result<&[u8]> {
        let len = cur.read_u32::<LittleEndian>()? as usize;
        let start = cur.position() as usize;
        let end = start + len;
        let s = &stream[start..end];
        cur.set_position(end as u64);
        Ok(s)
    };

    // Slice stream into block views
    let mut slices = Vec::with_capacity(blocks.len());
    for _ in &blocks {
        let residuals = read_prefixed(&mut cur)?;
        let means     = read_prefixed(&mut cur)?;
        let row_coefs = read_prefixed(&mut cur)?;
        let col_coefs = read_prefixed(&mut cur)?;
        slices.push(EncodedBlockSlice { residuals, means, row_coefs, col_coefs });
    }

    // Parallel decode 
    let decoded: Vec<((usize, usize), Array2<i32>)> = blocks
        .into_par_iter()
        .zip(slices.into_par_iter())
        .map(|((r, c), sl)| -> Result<((usize, usize), Array2<i32>)> {
            let r_end = cmp::min(r + p.block_height, h);
            let c_end = cmp::min(c + p.block_width,  w);
            let rows  = r_end - r;
            let cols  = c_end - c;
            let blk = bp
                .decode_block(sl, rows, cols)
                .with_context(|| format!("decode block @({r},{c})"))?;
            Ok(((r, c), blk))
        })
        .collect::<Result<_>>()?;

    // Single-threaded stitch
    let mut out = Array2::<i32>::zeros((h, w));
    for ((r, c), blk) in decoded {
        let r_end = cmp::min(r + p.block_height, h);
        let c_end = cmp::min(c + p.block_width,  w);
        let rows  = r_end - r;
        let cols  = c_end - c;
        // only copy the actual-size subview in case we’re at the edge
        out.slice_mut(s![r..r_end, c..c_end])
            .assign(&blk.slice(s![..rows, ..cols]));
    }

    Ok(out)
}
