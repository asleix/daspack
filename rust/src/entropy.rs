// rust/src/entropy.rs
// Fast residual compressor for DAS data (full source)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use constriction::stream::{stack::DefaultAnsCoder, Decode};
use constriction::{CoderError, DefaultEncoderFrontendError};
use constriction::stream::queue::DecoderFrontendError;
use numpy::ndarray::{Array2, Axis};
use std::convert::Infallible;
use std::io::{Cursor, Read};

pub mod exp_golomb;
mod fast_models;
mod mu_law_quant;
mod ans_tables;

use crate::entropy::fast_models::{FastGaussian, FastLaplace};
use crate::entropy::ans_tables::{gaussian_nz_bound, laplace_nz_bound};
use crate::entropy::mu_law_quant::MuLawQuantizer;


/* ────────────────────────────────────────────────────────────────────
 * Error handling
 * ────────────────────────────────────────────────────────────────── */

#[derive(thiserror::Error, Debug)]
pub enum CodecError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Range-coder error: {0}")]
    Range(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Encoder frontend error: {0}")]
    EncoderFrontend(#[from] DefaultEncoderFrontendError),
    #[error("Decoder frontend error: {0}")]
    DecoderFrontend(#[from] DecoderFrontendError),
    #[error("Corrupted bit-stream: {0}")]
    Corrupted(&'static str),
    #[error("Row too large: {0} > 2²² Bytes")]
    RowTooLarge(usize),
    #[error("Wrong Exp-Gol encoding shape")]
    ExpGolShape(#[from] numpy::ndarray::ShapeError),
}

impl From<CoderError<DefaultEncoderFrontendError, Infallible>> for CodecError {
    fn from(e: CoderError<DefaultEncoderFrontendError, Infallible>) -> Self {
        match e {
            CoderError::Frontend(fe) => CodecError::EncoderFrontend(fe),
            CoderError::Backend(infall) => match infall {}, // unreachable
        }
    }
}
impl From<CoderError<DecoderFrontendError, Infallible>> for CodecError {
    fn from(e: CoderError<DecoderFrontendError, Infallible>) -> Self {
        match e {
            CoderError::Frontend(fe) => CodecError::DecoderFrontend(fe),
            CoderError::Backend(infall) => match infall {}, // unreachable
        }
    }
}

type Result<T, E = CodecError> = std::result::Result<T, E>;

/* ────────────────────────────────────────────────────────────────────
 * Percentile – quick-select (mutable O(N))
 * ────────────────────────────────────────────────────────────────── */
fn percentile(arr: &[i32], pct: f64) -> i32 {
    if arr.is_empty() {
        return 0;
    }
    let mut tmp = arr.to_vec();
    let k = ((pct / 100.0) * (tmp.len() - 1) as f64).round() as usize;
    let (_, x, _) = tmp.select_nth_unstable(k);
    *x
}

/* ────────────────────────────────────────────────────────────────────
 * Scratch buffers reused for every row
 * ────────────────────────────────────────────────────────────────── */
struct RowScratch {
    clipped: Vec<i32>,
    diffs:   Vec<i32>,
    nz_cnt:  usize,
    cost: usize
}
impl RowScratch {
    fn with_capacity(cap: usize) -> Self {
        Self {
            clipped: Vec::with_capacity(cap),
            diffs:   Vec::with_capacity(cap),
            nz_cnt:  0,
            cost: 0
        }
    }
    #[inline]
    fn clear(&mut self) {
        self.clipped.clear();
        self.diffs.clear();
        self.nz_cnt = 0;
        self.cost = 0;
    }
}

/* ────────────────────────────────────────────────────────────────────
 * Row encoders
 * ────────────────────────────────────────────────────────────────── */

/// Laplace model – returns ANS words, μ-idx, σ-idx
fn encode_row_laplace(
    xs: &[i32],
    q: &MuLawQuantizer,
    scratch: &mut RowScratch,
) -> Result<(Vec<u32>, u8, u8)> {
    /* location & scale */
    let mu = percentile(xs, 50.0) as f64;
    let mad = xs
        .iter()
        .map(|&v| (v as f64 - mu).abs())
        .sum::<f64>()
        / xs.len() as f64;
    let b = mad / std::f64::consts::LN_2;

    let (mu_idx, mu_val) = q.quantize(mu as f32);
    let (sc_idx, sc_val) = q.quantize_ceil(b.max(1.0) as f32);

    /* clipping */
    let bound = laplace_nz_bound(sc_val as f64).max(0);
    let min_val = mu_val - bound;
    let max_val = mu_val + bound;
    let min_allowed = min_val - 1;
    let max_allowed = max_val + 1;

    scratch.clear();
    for &x in xs {
        let c = x.clamp(min_allowed, max_allowed);
        scratch.clipped.push(c);
        let d = x - c.clamp(min_val, max_val);
        if d != 0 {
            scratch.nz_cnt += 1;
        }
        scratch.diffs.push(d);
    }

    let mut dst: Vec<u8> = Vec::<u8>::new();
    write_row_tail(&scratch.diffs, & mut dst)?;

    scratch.cost = dst.len();

    let model = FastLaplace {
        sc_idx,
        mu: mu_val,
    };
    let mut ans = DefaultAnsCoder::new();
    ans.encode_iid_symbols_reverse(scratch.clipped.iter(), &model)?;
    Ok((ans.into_compressed().unwrap(), mu_idx, sc_idx))
}

/// Gaussian model – returns ANS words, μ-idx, σ-idx
fn encode_row_gaussian(
    xs: &[i32],
    q: &MuLawQuantizer,
    scratch: &mut RowScratch,
) -> Result<(Vec<u32>, u8, u8)> {
    /* location & scale */
    let mean = xs.iter().copied().sum::<i32>() as f64 / xs.len() as f64;
    let var = xs
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / xs.len() as f64;
    let stdev = var.sqrt().max(1.0);

    let (mu_idx, mu_val) = q.quantize(mean as f32);
    let (sc_idx, sc_val) = q.quantize_ceil(stdev as f32);

    /* clipping */
    let bound = gaussian_nz_bound(sc_val as f64).max(0);
    let min_val = mu_val - bound;
    let max_val = mu_val + bound;
    let min_allowed = min_val - 1;
    let max_allowed = max_val + 1;

    scratch.clear();
    for &x in xs {
        let c = x.clamp(min_allowed, max_allowed);
        scratch.clipped.push(c);
        let d = x - c.clamp(min_val, max_val);
        if d != 0 {
            scratch.nz_cnt += 1;
        }
        scratch.diffs.push(d);
    }

    let mut dst: Vec<u8> = Vec::<u8>::new();
    write_row_tail(&scratch.diffs, & mut dst)?;

    scratch.cost = dst.len();

    let model = FastGaussian {
        sc_idx,
        mu: mu_val,
    };
    let mut ans = DefaultAnsCoder::new();
    ans.encode_iid_symbols_reverse(scratch.clipped.iter(), &model)?;


    Ok((ans.into_compressed().unwrap(), mu_idx, sc_idx))
}

/// Raw Exp-Golomb model – returns payload bytes, k
fn encode_row_raw(xs: &[i32]) -> (Vec<u8>, u8) {
    let k = exp_golomb::estimate_best_k(xs);
    let bytes = exp_golomb::encode_k_expgolomb_list(xs, k);
    (bytes, k as u8)
}

/* ────────────────────────────────────────────────────────────────────
 * Public API – compressor
 * ────────────────────────────────────────────────────────────────── */

pub fn compress_residuals(res: &Array2<i32>) -> Result<Vec<u8>> {
    let (h, w) = res.dim();
    use crate::entropy::ans_tables::{SCALE_MU, SCALE_MAX};
    let q = MuLawQuantizer::new(SCALE_MU, SCALE_MAX);

    /* --- header -------------------------------------------------- */
    let mut out = Vec::<u8>::with_capacity(h * w * 2);
    out.write_i32::<LittleEndian>(h as i32)?;
    out.write_i32::<LittleEndian>(w as i32)?;

    /* cumulative sparse-tail diffs (Lap/Gau rows only) */
    let mut diffs_all = Vec::<i32>::with_capacity(h * w / 8);

    /* per-row scratch */
    let mut lap_scr = RowScratch::with_capacity(w);
    let mut gau_scr = RowScratch::with_capacity(w);

    for row in res.axis_iter(Axis(0)) {
        let xs = row.as_slice().expect("contiguous");

        /* ---------- encode three candidates ---------------------- */
        let (lap_words, lap_mu, lap_sc) = encode_row_laplace(xs, &q, &mut lap_scr)?;
        let (gau_words, gau_mu, gau_sc) = encode_row_gaussian(xs, &q, &mut gau_scr)?;
        let (raw_bytes, raw_k)          = encode_row_raw(xs);

        /* exact sizes in bytes (incl. overhead) ------------------- */
        let lap_bytes_tot = 1 + 2 + 3 + lap_words.len() * 4 + lap_scr.cost; //lap_scr.nz_cnt * NZ_EST_BITS;
        let gau_bytes_tot = 1 + 2 + 3 + gau_words.len() * 4 + gau_scr.cost; //gau_scr.nz_cnt * NZ_EST_BITS;
        let raw_bytes_tot = 1 + 1 + 3 + raw_bytes.len();

        /* choose smallest ---------------------------------------- */
        enum Model<'a> {
            Lap { words: &'a [u32], mu: u8, sc: u8, diffs: &'a [i32] },
            Gau { words: &'a [u32], mu: u8, sc: u8, diffs: &'a [i32] },
            Raw { bytes: &'a [u8],  k:  u8                       },
        }
        let model = if raw_bytes_tot <= lap_bytes_tot && raw_bytes_tot <= gau_bytes_tot {
            Model::Raw { bytes: &raw_bytes, k: raw_k }
        } else if lap_bytes_tot <= gau_bytes_tot {
            Model::Lap {
                words: &lap_words,
                mu:    lap_mu,
                sc:    lap_sc,
                diffs: &lap_scr.diffs,
            }
        } else {
            Model::Gau {
                words: &gau_words,
                mu:    gau_mu,
                sc:    gau_sc,
                diffs: &gau_scr.diffs,
            }
        };

        /* ---------- write chosen model --------------------------- */
        match model {
            Model::Lap { words, mu, sc, diffs } => {
                out.write_u8(1)?;              // flag
                out.write_u8(mu)?;
                out.write_u8(sc)?;
                write_ans_payload(words, &mut out)?;
                diffs_all.extend_from_slice(diffs);
            }
            Model::Gau { words, mu, sc, diffs } => {
                out.write_u8(0)?;              // flag
                out.write_u8(mu)?;
                out.write_u8(sc)?;
                write_ans_payload(words, &mut out)?;
                diffs_all.extend_from_slice(diffs);
            }
            Model::Raw { bytes, k } => {
                out.write_u8(2)?;              // flag
                out.write_u8(k)?;
                write_raw_payload(bytes, &mut out)?;
            }
        }
    }

    /* ---------- global sparse-tail (for Lap/Gau rows) ----------- */
    let non_zero: Vec<i32> = diffs_all.into_iter().filter(|&d| d != 0).collect();
    let k = exp_golomb::estimate_best_k(&non_zero);
    out.write_u8(k as u8)?;
    out.extend(exp_golomb::encode_k_expgolomb_list(&non_zero, k));

    Ok(out)
}

/* helpers */
#[inline]
fn write_row_tail(diffs: &[i32], dst: &mut Vec<u8>) -> Result<()> {
    let nz: Vec<i32> = diffs.iter().copied().filter(|&d| d != 0).collect();
    let k = exp_golomb::estimate_best_k(&nz);
    dst.write_u8(k as u8)?;
    dst.extend(exp_golomb::encode_k_expgolomb_list(&nz, k));
    Ok(())
}

#[inline]
fn write_ans_payload(words: &[u32], dst: &mut Vec<u8>) -> Result<()> {
    if words.len() > 0xFF_FFFF {
        return Err(CodecError::RowTooLarge(words.len() << 2));
    }
    let len24 = (words.len() as u32).to_le_bytes();
    dst.extend_from_slice(&len24[..3]);
    for &w in words {
        dst.extend_from_slice(&w.to_le_bytes());
    }
    Ok(())
}
#[inline]
fn write_raw_payload(bytes: &[u8], dst: &mut Vec<u8>) -> Result<()> {
    if bytes.len() > 0xFF_FFFF {
        return Err(CodecError::RowTooLarge(bytes.len()));
    }
    let len24 = (bytes.len() as u32).to_le_bytes();
    dst.extend_from_slice(&len24[..3]);
    dst.extend_from_slice(bytes);
    Ok(())
}

use ndarray::ArrayView2;
pub fn compress_residuals_inplace(res: ArrayView2<'_, i32>) -> Result<Vec<u8>> {
    let (h, w) = res.dim();

    /* ---- header ------------------------------------------------ */
    let mut out = Vec::<u8>::with_capacity(h * w / 2);
    out.write_i32::<LittleEndian>(h as i32)?;
    out.write_i32::<LittleEndian>(w as i32)?;

    

    /* ---- per-row payloads -------------------------------------- */
    for row in res.axis_iter(Axis(0)) {
        let xs = row
            .as_slice()
            .expect("row view must be contiguous (C-order)");

        /* raw Exp-Golomb for the whole row */
        let (bytes, k) = encode_row_raw(xs);

        /* flag = 2 → raw, followed by k and payload */
        out.write_u8(2)?;      // model flag
        out.write_u8(k)?;      // Exp-Golomb k
        write_raw_payload(&bytes, &mut out)?;   // 24-bit length + bytes
    }

    /* ---- global sparse-tail (unused) --------------------------- */
    out.write_u8(0)?;          // k = 0  (no values follow)
    /* no bytes follow because there are no Lap/Gau corrections */

    Ok(out)
}


/* ────────────────────────────────────────────────────────────────────
 * Decompressor
 * ────────────────────────────────────────────────────────────────── */

pub fn decompress_residuals(buf: &[u8]) -> Result<Array2<i32>> {
    let mut cur = Cursor::new(buf);
    let h = cur.read_i32::<LittleEndian>()? as usize;
    let w = cur.read_i32::<LittleEndian>()? as usize;
    use crate::entropy::ans_tables::{SCALE_MU, SCALE_MAX};
    let q = MuLawQuantizer::new(SCALE_MU, SCALE_MAX);

    let mut out = Array2::<i32>::zeros((h, w));
    let mut corrections = Vec::<(usize, usize)>::new();

    let mut scratch_u32 = Vec::<u32>::new(); // one allocation
    for r in 0..h {
        let flag = cur.read_u8()?;

        match flag {
            /* -------- Gaussian ----------------------------------- */
            0 | 1 => {
                let use_lap = flag == 1;
                let mu_idx  = cur.read_u8()?;
                let sc_idx  = cur.read_u8()?;
                let mu_val  = q.dequantize(mu_idx) as f64;
                let sc_val  = q.dequantize(sc_idx).max(1) as f64;

                /* clipping range */
                let bound = if use_lap {
                    laplace_nz_bound(sc_val)
                } else {
                    gaussian_nz_bound(sc_val)
                };
                let min_val     = mu_val as i32 - bound;
                let max_val     = mu_val as i32 + bound;
                let min_allowed = min_val - 1;
                let max_allowed = max_val + 1;

                /* payload length */
                let bytes = (read_len24(&mut cur)? as usize) * 4;
                let slice_u8 = take_slice(buf, &mut cur, bytes)?;

                /* u8→u32 view */
                // if (slice_u8.as_ptr() as usize) & 3 == 0 {
                //     bytemuck::cast_slice(slice_u8)
                // } else
                let slice_u32: &[u32] =  {
                    scratch_u32.clear();
                    scratch_u32.extend(
                        slice_u8
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                    );
                    &scratch_u32
                };

                let mut ans =
                    DefaultAnsCoder::from_compressed_slice(slice_u32).expect("corrupted ANS row");
                let mut row = out.row_mut(r);
                let dst = row.as_slice_mut().unwrap();

                if use_lap {
                    let model = FastLaplace {
                        sc_idx,
                        mu: mu_val as i32,
                    };
                    for (c, res) in ans.decode_iid_symbols(w, &model).enumerate() {
                        let v = res.unwrap();
                        dst[c] = v;
                        if v == min_allowed || v == max_allowed {
                            corrections.push((r, c));
                        }
                    }
                } else {
                    let model = FastGaussian {
                        sc_idx,
                        mu: mu_val as i32,
                    };
                    for (c, res) in ans.decode_iid_symbols(w, &model).enumerate() {
                        let v = res.unwrap();
                        dst[c] = v;
                        if v == min_allowed || v == max_allowed {
                            corrections.push((r, c));
                        }
                    }
                }


            }

            /* -------- Raw Exp-Golomb ------------------------------ */
            2 => {
                let k = cur.read_u8()? as u32;
                let bytes = read_len24(&mut cur)? as usize;
                let slice = take_slice(buf, &mut cur, bytes)?;
                let vals = exp_golomb::decode_k_expgolomb_list(slice, w, k);
                out.row_mut(r)
                    .as_slice_mut()
                    .expect("row not contiguous")
                    .copy_from_slice(&vals);
            }

            _ => return Err(CodecError::Corrupted("unknown model flag")),
        }
    }

    /* ---------- tail corrections (Lap/Gau rows only) ------------- */
    let k = cur.read_u8()? as u32;
    let mut remaining = Vec::new();
    cur.read_to_end(&mut remaining)?;
    let corr_vals =
        exp_golomb::decode_k_expgolomb_list(&remaining, corrections.len(), k);

    for (idx, (r, c)) in corrections.into_iter().enumerate() {
        let v = corr_vals[idx];
        let sgn = v.signum();
        out[[r, c]] += v - sgn; // inverse of mapping during encode
    }

    Ok(out)
}

/* helpers */
#[inline]
fn read_len24(cur: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut b = [0u8; 4];
    cur.read_exact(&mut b[..3])?;
    Ok(u32::from_le_bytes(b))
}
#[inline]
fn take_slice<'a>(
    buf: &'a [u8],
    cur: &mut Cursor<&[u8]>,
    bytes: usize,
) -> Result<&'a [u8]> {
    let start = cur.position() as usize;
    let end = start + bytes;
    if end > buf.len() {
        return Err(CodecError::Corrupted("payload overrun"));
    }
    cur.set_position(end as u64);
    Ok(&buf[start..end])
}

/* ────────────────────────────────────────────────────────────────────
 * Tests
 * ────────────────────────────────────────────────────────────────── */
#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn roundtrip_fixed() {
        let data = array![
            [10, 1000, -200, 10],
            [500, 501, -30000, 32000],
            [0, 0, 0, 0],
        ];
        let bytes = compress_residuals(&data).unwrap();
        let recon = decompress_residuals(&bytes).unwrap();
        assert_eq!(data, recon);
    }

    #[test]
    fn roundtrip_random() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0xDADBEEF);
        let res: Array2<i32> = Array2::from_shape_fn((128, 64), |_| rng.random_range(-32768..32768));
        let bytes = compress_residuals(&res).unwrap();
        let recon = decompress_residuals(&bytes).unwrap();
        assert_eq!(res, recon);
    }
}
