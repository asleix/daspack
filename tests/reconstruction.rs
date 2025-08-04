//! Round-trip tests for residual and lossless compressors.
//!
//! We generate synthetic integer 2D data (uniform & Gaussian-like),
//! then verify that decompress(compress(data)) == data for:
//!   - compress_residuals  / decompress_residuals
//!   - compress_lossless   / decompress_lossless
//!
//! The RNG uses fixed seeds to make failures reproducible.
//!
//! Dev-deps (add to Cargo.toml if needed):
//! ```toml
//! [dev-dependencies]
//! ndarray = "0.15"
//! rand = "0.8"
//! rand_distr = "0.4"
//! ```

use compute::{
    compress_lossless, decompress_lossless, compress_residuals_rice, decompress_residuals_rice, CompressParams,
};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Helper: construct an Array2<i32> by filling with a closure.
fn array2_from_fn<F>(rows: usize, cols: usize, mut f: F) -> Array2<i32>
where
    F: FnMut(usize, usize) -> i32,
{
    ndarray::Array2::from_shape_fn((rows, cols), |(r, c)| f(r, c))
}

/// Uniform integer data in [lo, hi] inclusive.
fn make_uniform(rows: usize, cols: usize, lo: i32, hi: i32, rng: &mut StdRng) -> Array2<i32> {
    assert!(lo <= hi);
    array2_from_fn(rows, cols, |_r, _c| rng.random_range(lo..=hi))
}

/// Gaussian-like integer data (rounded), with mean & stddev.
/// Values are rounded to nearest i32 and clamped to i32 bounds.
fn make_gaussian(rows: usize, cols: usize, mean: f64, stddev: f64, rng: &mut StdRng) -> Array2<i32> {
    let normal = Normal::new(mean, stddev).expect("valid normal params");
    array2_from_fn(rows, cols, |_r, _c| {
        let v = normal.sample(rng).round();
        v.clamp(i32::MIN as f64, i32::MAX as f64) as i32
    })
}

/// Residuals round trip for a given matrix.
fn roundtrip_residuals(data: &Array2<i32>) {
    let bytes = compress_residuals_rice(data.view()).expect("compress_residuals should succeed");
    let decoded = decompress_residuals_rice(&bytes).expect("decompress_residuals should succeed");
    assert_eq!(data, &decoded, "Residual round-trip mismatch");
}

/// Lossless round trip for a given matrix and parameters.
/// Tests both `row_demean` settings for the same parameter set.
fn roundtrip_lossless(data: &Array2<i32>, mut p: CompressParams) {
    for &row_demean in &[true, false] {
        p.row_demean = row_demean;

        let bytes =
            compress_lossless(data, p.clone()).expect("compress_lossless should succeed");
        let decoded = decompress_lossless(&bytes, data.dim(), p.clone())
            .expect("decompress_lossless should succeed");

        assert_eq!(
            data, &decoded,
            "Lossless round-trip mismatch (row_demean={:?}, params={:?})",
            row_demean, p
        );
    }
}

#[test]
fn residuals_roundtrip_uniform_and_gaussian() {
    // Shapes include edge cases and non-multiples of common block sizes.
    let shapes = [
        (1, 1),
        (1, 17),
        (17, 1),
        (16, 16),
        (31, 63),
        (64, 64),
        (127, 35),
    ];

    // Fixed seed for reproducibility.
    let mut rng = StdRng::seed_from_u64(0x5EED_u64);

    for &(h, w) in &shapes {
        // Uniform: moderate range exercises code paths while avoiding extremes.
        let uni = make_uniform(h, w, -2000, 2000, &mut rng);
        roundtrip_residuals(&uni);

        // Gaussian-like: mean near zero, moderate stddev.
        let gau = make_gaussian(h, w, 0.0, 300.0, &mut rng);
        roundtrip_residuals(&gau);
    }
}

#[test]
fn lossless_roundtrip_various_params_and_data() {
    let shapes = [
        (8, 8),
        (13, 29),
        (32, 16),
        (48, 48),
        (65, 33),
        (127, 35),
    ];

    // Parameter grid: (block_h, block_w, lx, lt, lpc_order, tail_cut).
    let param_grid = [
        (16, 16, 1, 1, 4, 5.0),
        (32, 16, 1, 1, 4, 6.0),
        (16, 32, 1, 1, 6, 6.0),
    ];

    let mut rng = StdRng::seed_from_u64(42);

    for &(h, w) in &shapes {
        // Build two datasets per shape: uniform & gaussian-like.
        let data_uniform = make_uniform(h, w, -1500, 1500, &mut rng);
        let data_gauss = make_gaussian(h, w, 10.0, 200.0, &mut rng);

        for &(bh, bw, lx, lt, lpc_order, tail_cut) in &param_grid {
            let p = CompressParams::new(bh, bw, lx, lt, lpc_order, tail_cut);

            roundtrip_lossless(&data_uniform, p.clone());
            roundtrip_lossless(&data_gauss, p.clone());
        }
    }
}
