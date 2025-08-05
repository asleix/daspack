// prediction.rs

use ndarray::{Array2, s};

use super::wavelets;
use super::lpctool::LpcTool;


// -----------------------------------------------------------------------
// Multi-block predictor using wavelet transforms + LPC
// -----------------------------------------------------------------------

pub struct MultiBlockPredictor {
    pub lx: usize,
    pub lt: usize,
    pub lpc_tool: LpcTool,
}

impl MultiBlockPredictor {
    pub fn new(lx: usize, lt: usize, order: usize, nbits: u8, lpc_range: (f64, f64)) -> Self {
        let lpc_tool = LpcTool::new(order, nbits, lpc_range.1, lpc_range.0);
        Self { lx, lt, lpc_tool}
    }

    /// Forward prediction for 2D data (in an `Array2<i32>`):
    ///  1) Forward wavelet transform
    ///  2) Row-wise LPC on LL subband (top-left portion)
    ///  3) Column-wise LPC on LL subband
    ///
    /// Returns:
    ///   ( transformed_data, row_coefs (n x (order+1)), col_coefs (m x (order+1)) )
    pub fn predict_diff(&self, data: &Array2<i32>) -> (Array2<i32>, Array2<f64>, Array2<f64>) {
        let (n, m) = data.dim();
        // Copy data to transform in-place
        let mut txfm = data.clone();

        // 1) Forward wavelet transforms
       wavelets::fwd_txfm2d_levels_inplace(&mut txfm, self.lx);

        // The "extent" of the LL subband
        let row_extent = m >> self.lt; // columns of LL
        let col_extent = n >> self.lx; // rows of LL

        // We'll store row and column LPC coefficients in 2D arrays
        let lpc_order = self.lpc_tool.order;
        let mut row_coefs_list = Array2::<f64>::zeros((n, lpc_order + 1));
        let mut col_coefs_list = Array2::<f64>::zeros((m, lpc_order + 1));

        // 2) Row-wise LPC on first `row_extent` columns of each row
        for i in 0..n {
            // Grab the portion of row i that belongs to LL
            let row_slice = txfm.slice(s![i, 0..row_extent]);
            // Convert that slice to Vec<i32> for the LPC tool
            let row_vec = row_slice.to_vec();
            // Compute & quantize LPC coefs, plus residual
            let (a, residual) = self.lpc_tool.get_coefs_and_residuals(&row_vec);

            // Store the coefs in row_coefs_list
            for (k, &val) in a.iter().enumerate() {
                row_coefs_list[[i, k]] = val;
            }
            // Write the residual back into the transform
            for (idx, &val) in residual.iter().enumerate() {
                txfm[[i, idx]] = val;
            }
        }

        // 3) Column-wise LPC on first `col_extent` rows of each column
        for j in 0..m {
            let col_slice = txfm.slice(s![0..col_extent, j]);
            let col_vec = col_slice.to_vec();
            let (a, residual) = self.lpc_tool.get_coefs_and_residuals(&col_vec);

            // Store coefs
            for (k, &val) in a.iter().enumerate() {
                col_coefs_list[[j, k]] = val;
            }
            // Put the residual back
            for (i, &val) in residual.iter().enumerate() {
                txfm[[i, j]] = val;
            }
        }

        (txfm, row_coefs_list, col_coefs_list)
    }

    /// Inverse reconstruction:
    ///  1) Column-wise inverse LPC
    ///  2) Row-wise inverse LPC
    ///  3) Inverse wavelet transforms
    ///
    /// Returns the reconstructed 2D array.
    pub fn reconstruct_diff(
        &self,
        mut txfm: Array2<i32>,
        row_coefs_list: &Array2<f64>,
        col_coefs_list: &Array2<f64>,
    ) -> Array2<i32> {
        let (n, m) = txfm.dim();

        let row_extent = m >> self.lt;
        let col_extent = n >> self.lx;

        //let lpc_tool = LpcTool::new(self.lpc_order, 6, 1.0, -1.0);

        // 1) Column-wise decode
        for j in 0..m {
            // Extract the residual from LL portion in column j
            let mut residual = Vec::with_capacity(col_extent);
            for i in 0..col_extent {
                residual.push(txfm[[i, j]]);
            }
            // The column's LPC coefs
            let a = col_coefs_list.slice(s![j, ..]).to_vec();
            // Decode
            let rec_col = self.lpc_tool.decode_lpc(&residual, &a);
            // Put it back
            for (i, &val) in rec_col.iter().enumerate() {
                txfm[[i, j]] = val;
            }
        }

        // 2) Row-wise decode
        for i in 0..n {
            let mut residual = Vec::with_capacity(row_extent);
            for idx in 0..row_extent {
                residual.push(txfm[[i, idx]]);
            }
            let a = row_coefs_list.slice(s![i, ..]).to_vec();
            let rec_row = self.lpc_tool.decode_lpc(&residual, &a);

            for (idx, &val) in rec_row.iter().enumerate() {
                txfm[[i, idx]] = val;
            }
        }

        // 3) Inverse wavelet transforms
        wavelets::inv_txfm2d_levels_inplace(&mut txfm, self.lx);

        txfm
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2};
    use rand::{SeedableRng};
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal, Uniform};
    use ndarray::array;

    #[test]
    fn test_prediction_reconstruction() {
        // Example: 4x4 data stored in an Array2
        let data = array![
            [10, 11, 12, 13],
            [14, 15, 16, 17],
            [20, 21, 22, 23],
            [24, 25, 26, 27]
        ];

        let predictor = MultiBlockPredictor::new(1, 1, 1, 6, (-1.5, 1.5));

        let (txfm, row_coefs, col_coefs) = predictor.predict_diff(&data);
        let reconstructed = predictor.reconstruct_diff(txfm, &row_coefs, &col_coefs);

        assert_eq!(reconstructed, data);
    }

    // ---------- helpers ----------

    fn gen_uniform_i32(n: usize, m: usize, seed: u64, lo: i32, hi: i32) -> Array2<i32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Uniform::<i32>::new_inclusive(lo, hi)
            .expect("invalid Uniform bounds");
        Array2::from_shape_fn((n, m), |_| dist.sample(&mut rng))
    } 

     fn gen_normal_i32(n: usize, m: usize, seed: u64, mean: f64, std: f64, clamp: i32) -> Array2<i32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(mean, std).unwrap();
        Array2::from_shape_fn((n, m), |_| {
            let v = dist.sample(&mut rng);
            v.round()
                .clamp(-(clamp as f64), clamp as f64) as i32
        })
    }

    /// Small sanity to keep LL extents >= order+1.
    /// row_extent = m >> lt, col_extent = n >> lx; both must be > order.
    fn max_safe_order(n: usize, m: usize, lx: usize, lt: usize, hard_cap: usize) -> usize {
        let row_extent = m >> lt;
        let col_extent = n >> lx;
        row_extent.min(col_extent).saturating_sub(1).min(hard_cap).max(1)
    }

    // ---------- tests ----------

    #[test]
    fn roundtrip_random_uniform_various_sizes_orders() {
        // Choose sizes divisible by 2^lx and 2^lt. Keep lx == lt to match current impl.
        let sizes = &[(8, 8), (8, 16), (16, 8), (16, 16), (32, 32)];
        let bit_depths = &[6u8, 8u8, 10u8];
        let lpc_ranges = &[(-1.5, 1.5), (-1.9, 1.9)];
        for &(n, m) in sizes {
            for &lx in &[1usize, 2usize] {
                let lt = lx; // keep equal to avoid extents mismatch assumptions
                // Keep order strictly < min(row_extent, col_extent)
                let max_order = max_safe_order(n, m, lx, lt, 4);
                for order in 1..=max_order {
                    for &nbits in bit_depths {
                        for &range in lpc_ranges {
                            for seed in 0u64..3 {
                                let data = gen_uniform_i32(n, m, seed, -5000, 5000);
                                let predictor = MultiBlockPredictor::new(lx, lt, order, nbits, range);
                                let (txfm, row_coefs, col_coefs) = predictor.predict_diff(&data);

                                // coefficient shapes
                                assert_eq!(row_coefs.dim(), (n, order + 1));
                                assert_eq!(col_coefs.dim(), (m, order + 1));

                                // lossless round-trip
                                let rec = predictor.reconstruct_diff(txfm, &row_coefs, &col_coefs);
                                assert_eq!(
                                    rec, data,
                                    "Mismatch: n={n}, m={m}, lx={lx}, order={order}, nbits={nbits}, range={range:?}, seed={seed}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn roundtrip_random_normal_heavy_values() {
        // Heavier-tailed-ish (via larger std) normal data; clamp prevents i32 overflow.
        let cases = &[
            (8, 8, 1usize),
            (16, 16, 2usize),
            (32, 32, 2usize),
        ];
        for &(n, m, lx) in cases {
            let lt = lx;
            let max_order = max_safe_order(n, m, lx, lt, 3);
            for order in 1..=max_order {
                for seed in 10u64..13 {
                    let data = gen_normal_i32(n, m, seed, 0.0, 1500.0, 50_000);
                    let predictor = MultiBlockPredictor::new(lx, lt, order, 8, (-1.8, 1.8));
                    let (txfm, row_coefs, col_coefs) = predictor.predict_diff(&data);
                    let rec = predictor.reconstruct_diff(txfm, &row_coefs, &col_coefs);
                    assert_eq!(rec, data, "Normal roundtrip failed (n={n}, m={m}, order={order}, seed={seed})");
                }
            }
        }
    }

    #[test]
    fn roundtrip_rectangular_and_constant_inputs() {
        // Rectangular
        let (n, m, lx, lt) = (16, 32, 2usize, 2usize);
        let order = max_safe_order(n, m, lx, lt, 3);
        let predictor = MultiBlockPredictor::new(lx, lt, order, 6, (-1.5, 1.5));

        // Rectangular uniform random
        let data_u = gen_uniform_i32(n, m, 123, -1000, 1000);
        let (txfm_u, row_u, col_u) = predictor.predict_diff(&data_u);
        let rec_u = predictor.reconstruct_diff(txfm_u, &row_u, &col_u);
        assert_eq!(rec_u, data_u);

        // Constant matrix
        let val = 7i32;
        let data_c = Array2::from_elem((n, m), val);
        let (txfm_c, row_c, col_c) = predictor.predict_diff(&data_c);
        let rec_c = predictor.reconstruct_diff(txfm_c, &row_c, &col_c);
        assert_eq!(rec_c, data_c);
    }

    #[test]
    fn idempotent_multiple_roundtrips() {
        // Apply predict/reconstruct twice; should remain stable.
        let (n, m, lx, lt, order) = (16, 16, 2usize, 2usize, 2usize);
        let predictor = MultiBlockPredictor::new(lx, lt, order, 6, (-1.5, 1.5));
        let data = gen_uniform_i32(n, m, 999, -2048, 2048);

        // 1st round-trip
        let (tx1, r1, c1) = predictor.predict_diff(&data);
        let rec1 = predictor.reconstruct_diff(tx1, &r1, &c1);
        assert_eq!(rec1, data);

        // 2nd round-trip on the reconstructed data
        let (tx2, r2, c2) = predictor.predict_diff(&rec1);
        let rec2 = predictor.reconstruct_diff(tx2, &r2, &c2);
        assert_eq!(rec2, data);
    }
}
