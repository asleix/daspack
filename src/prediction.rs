// prediction.rs

use ndarray::{Array2, s};

use crate::wavelets;

/// A tool for Linear Predictive Coding (LPC) operations.
#[derive(Debug, Clone)]
pub struct LpcTool {
    pub order: usize,
    pub nbits: u8,
    pub max_coef: f64,
    pub min_coef: f64,
}

impl LpcTool {
    pub fn new(order: usize, nbits: u8, max_coef: f64, min_coef: f64) -> Self {
        LpcTool {
            order,
            nbits,
            max_coef,
            min_coef,
        }
    }

    pub fn quantize_uniform(&self, value: f64) -> u32 {
        let clipped = value.clamp(self.min_coef, self.max_coef);
        let levels = (1 << self.nbits) - 1;
        let step = (self.max_coef - self.min_coef) / (levels as f64);
        let index = ((clipped - self.min_coef) / step).round();
        if index < 0.0 { 0 } else { index as u32 }
    }

    pub fn dequantize_uniform(&self, index: u32) -> f64 {
        let levels = (1 << self.nbits) - 1;
        let step = (self.max_coef - self.min_coef) / (levels as f64);
        self.min_coef + (index as f64) * step
    }

    pub fn _get_symbols(&self, coefs: &[f64]) -> Vec<u32> {
        coefs.iter().map(|&c| self.quantize_uniform(c)).collect()
    }

    pub fn coefs_from_symbols(&self, symbols: &[u32]) -> Vec<f64> {
        symbols
            .iter()
            .map(|&sym| self.dequantize_uniform(sym))
            .collect()
    }

    pub fn quantize_coefficients(&self, coefs: &[f64]) -> Vec<f64> {
        coefs
            .iter()
            .map(|&c| {
                let sym = self.quantize_uniform(c);
                self.dequantize_uniform(sym)
            })
            .collect()
    }

    /// Compute autocorrelation up to `order` lags.
    fn autocorrelation(x: &[f64], order: usize) -> Vec<f64> {
        let n = x.len();
        let mut r = vec![0.0; order + 1];
        for lag in 0..=order {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += x[i] * x[i + lag];
            }
            r[lag] = sum;
        }
        r
    }

    /// Levinson–Durbin recursion to get LPC coefficients.
    fn levinson_durbin(r: &[f64], order: usize) -> Vec<f64> {
        let mut a = vec![0.0; order + 1];
        a[0] = 1.0;
        let mut e = r[0];

        for i in 1..=order {
            let mut acc = 0.0;
            for j in 1..i {
                acc += a[j] * r[i - j];
            }

            if e.abs() < 1e-12{ break; }
            let k = (r[i] - acc) / e;

            for j in 1..i {
                a[j] -= k * a[i - j];
            }
            a[i] = k;
            e *= 1.0 - k * k;
        }
        a
    }

    /// Compute standard LPC coefficients (a[0..=order]) from a signal.
    /// a[0] = 1.0, then a[1..] = - (Levinson–Durbin reflection).
    fn lpc(&self, signal: &[i32]) -> Vec<f64> {
        let signal_f: Vec<f64> = signal.iter().map(|&v| v as f64).collect();
        let r = Self::autocorrelation(&signal_f, self.order);
        let mut a = Self::levinson_durbin(&r, self.order);

        for coef in &mut a[1..] {
            *coef = -*coef;
        }
        a
    }

    /// Predict using LPC coefficients.
    fn predict(&self, signal: &[i32], a: &[f64]) -> Vec<i32> {
        let mut predicted = vec![0i32; signal.len()];
        let order = self.order.min(a.len().saturating_sub(1));

        for i in order..signal.len() {
            let mut sum_val = 0.0;
            for k in 1..=order {
                sum_val += a[k] * (signal[i - k] as f64);
            }
            predicted[i] = -(sum_val.floor() as i32);
        }
        predicted
    }

    /// Encode signal as residual = original - predicted.
    fn encode_lpc(&self, signal: &[i32], a: &[f64]) -> Vec<i32> {
        let predicted = self.predict(signal, a);
        signal
            .iter()
            .zip(predicted.iter())
            .map(|(orig, pred)| orig - pred)
            .collect()
    }

    /// Decode from residual using LPC coefficients.
    fn decode_lpc(&self, residual: &[i32], a: &[f64]) -> Vec<i32> {
        let n = residual.len();
        let order = self.order.min(a.len().saturating_sub(1));
        let mut rec_signal = vec![0i32; n];

        let limit = order.min(n);
        rec_signal[..limit].copy_from_slice(&residual[..limit]);
        // for i in 0..limit {
        //     rec_signal[i] = residual[i];
        // }
        for i in order..n {
            let mut sum_val = 0.0;
            for k in 1..=order {
                sum_val += a[k] * (rec_signal[i - k] as f64);
            }
            let inc = sum_val.floor() as i32;
            rec_signal[i] = residual[i] - inc;
        }
        rec_signal
    }

    /// Compute LPC coefs, quantize them, and return (quantized coefs, residual).
    pub fn get_coefs_and_residuals(&self, signal: &[i32]) -> (Vec<f64>, Vec<i32>) {
        let mut a = self.lpc(signal);
        let a_tail = self.quantize_coefficients(&a[1..]);
        for (dst, src) in a[1..].iter_mut().zip(a_tail.iter()) {
            *dst = *src;
        }
        let residual = self.encode_lpc(signal, &a);
        (a, residual)
    }
}

// -----------------------------------------------------------------------
// Multi-block predictor using wavelet transforms + LPC
// -----------------------------------------------------------------------

pub struct MultiBlockPredictor {
    pub lx: usize,
    pub lt: usize,
    pub lpc_tool: LpcTool,
}

impl MultiBlockPredictor {
    pub fn new(lx: usize, lt: usize, lpc_tool: LpcTool) -> Self {
        Self { lx, lt, lpc_tool }
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

// ---------------------------------------------------------
// Example test (adapted) showing usage with Array2
// ---------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
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

        let lpc_tool = LpcTool::new(1, 6, 1.0, -1.0);
        let predictor = MultiBlockPredictor::new(1, 1, lpc_tool);

        let (txfm, row_coefs, col_coefs) = predictor.predict_diff(&data);
        let reconstructed = predictor.reconstruct_diff(txfm, &row_coefs, &col_coefs);

        assert_eq!(reconstructed, data);
    }
}
