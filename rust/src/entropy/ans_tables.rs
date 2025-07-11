// rust/src/entropy/ans_tables.rs
//! Runtime-initialised CDF tables (Laplace & Gaussian) for the fast
//! integer models.  Heap allocated ⇒ no huge stack frame, safe on all
//! targets.

use once_cell::sync::Lazy;
use libm::erf;
use crate::entropy::mu_law_quant::MuLawQuantizer;
use std::num::NonZeroU16;
use std::f64::consts::PI;

/* ───────────── constants ────────────────────────────────────────── */
pub const PREC:      usize = 12;                // fixed-point bits
pub const SCALE:     usize = 1 << PREC;         // 4096
pub const MAX_VAL:   i32   = SCALE as i32;      // +- 4096 indices
pub const SCALES:    usize = 256;               // 1-byte σ index
pub const TABLE_LEN: usize = (MAX_VAL as usize) * 2 + 2;

/* ───────────── analytic “non-zero” bounds ───────────────────────── */
#[inline]
pub fn laplace_nz_bound(b: f64) -> i32 {
    let q = (-1.0 / b).exp();
    let rhs = 0.5 / SCALE as f64;
    let bound = ((rhs * (1.0 + q)) / (1.0 - q)).ln() / q.ln();
    bound.ceil() as i32
}

#[inline]
pub fn gaussian_nz_bound(sigma: f64) -> i32 {
    let lhs = 0.5 / SCALE as f64 * sigma * (2.0 * PI).sqrt();
    let bound_sq = -2.0 * sigma * sigma * lhs.ln();
    bound_sq.sqrt().ceil() as i32
}

/* ───────────── μ-law de-quantisation of σ/b ─────────────────────── */

pub const SCALE_MU  : f32 = 255.0; //255.0;
pub const SCALE_MAX : f32 = 16_383.0;

#[inline]
fn dequantize_scale(sc_idx: u8) -> f64 {
    // identical to codec’s MuLawQuantizer settings
    let q = MuLawQuantizer::new(SCALE_MU, SCALE_MAX);
    q.dequantize(sc_idx) as f64
}

/* ───────────── PMF helpers (bin-integrated) ─────────────────────── */
#[inline]
fn laplace_pmf(k: i32, b: f64) -> f64 {
    let a = k.abs() as f64;
    (-(a - 0.5) / b).exp() - (-(a + 0.5) / b).exp()
}

#[inline]
fn gaussian_pmf(k: i32, sigma: f64) -> f64 {
    const SQRT_2: f64 = std::f64::consts::SQRT_2;
    0.5 * (erf((k as f64 + 0.5) / (sigma * SQRT_2))
         - erf((k as f64 - 0.5) / (sigma * SQRT_2)))
}

/* ───────────── generic row builder (largest remainder) ───────────── */
fn build_row<F>(sc_idx: u8, pmf_fn: F, bound_fn: fn(f64) -> i32) -> [u16; TABLE_LEN]
where
    F: Fn(i32, f64) -> f64,
{
    let s = dequantize_scale(sc_idx).max(1.0);

    // include the two “overflow” bins at ±(bound+1)
    let mut bound = bound_fn(s) + 1;
    if bound > MAX_VAL {
        bound = MAX_VAL;
    }

    // if the support would exceed SCALE, shrink symmetrically
    let mut support_len = 2 * bound + 1;
    if support_len as usize > SCALE {
        bound = (SCALE as i32 - 1) / 2;
        support_len = 2 * bound + 1;
    }

    /* ----- gather PMF over the chosen support -------------------- */
    let mut probs_f = vec![0f64; support_len as usize];
    let mut total = 0.0;
    for (i, k) in (-bound..=bound).enumerate() {
        let p = pmf_fn(k, s);
        probs_f[i] = p;
        total += p;
    }
    // renormalise
    for p in &mut probs_f {
        *p /= total;
    }

    /* ----- quantise to u16 with “≥1-ulp” guarantee --------------- */
    let mut ints = vec![1u16; probs_f.len()];          // start with 1 each
    let mut remaining = SCALE - probs_f.len();         // tokens still to distribute

    // sort indices by descending probability
    let mut idxs: Vec<_> = (0..probs_f.len()).collect();
    idxs.sort_by(|&a, &b| probs_f[b].partial_cmp(&probs_f[a]).unwrap());

    let mut cursor = 0;
    while remaining > 0 {
        let i = idxs[cursor];
        ints[i] += 1;
        remaining -= 1;
        cursor = (cursor + 1) % idxs.len();
    }

    /* ----- build cumulative row ---------------------------------- */
    let mut row = [0u16; TABLE_LEN];
    let mut cum = 0u16;
    let mut support_iter = ints.into_iter();
    for (i, k) in (-MAX_VAL..=MAX_VAL).enumerate() {
        if k.abs() <= bound {
            cum = cum.saturating_add(support_iter.next().unwrap());
        }
        row[i + 1] = cum;
    }
    debug_assert_eq!(cum as usize, SCALE);             // exactly normalised
    row
}

/* ───────────── concrete tables (lazy-initialised) ───────────────── */
fn laplace_row(sc_idx: u8) -> [u16; TABLE_LEN] {
    build_row(sc_idx, laplace_pmf, laplace_nz_bound)
}
fn gaussian_row(sc_idx: u8) -> [u16; TABLE_LEN] {
    build_row(sc_idx, gaussian_pmf, gaussian_nz_bound)
}

fn build_table(row_fn: fn(u8) -> [u16; TABLE_LEN]) -> Box<[[u16; TABLE_LEN]]> {
    (0..SCALES).map(|sc| row_fn(sc as u8)).collect()
}

pub static LAPLACE_CDF:  Lazy<Box<[[u16; TABLE_LEN]]>> =
    Lazy::new(|| build_table(laplace_row));
pub static GAUSSIAN_CDF: Lazy<Box<[[u16; TABLE_LEN]]>> =
    Lazy::new(|| build_table(gaussian_row));

/* ───────────── helper for NonZeroU16 ------------------------------ */
#[inline(always)]
pub fn nz(p: u16) -> NonZeroU16 {
    unsafe { NonZeroU16::new_unchecked(p) }    // caller guarantees p > 0
}
