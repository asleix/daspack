//! src/core/lpctool.rs

/// A tool for Linear Predictive Coding (LPC) operations.
/// All public methods are panic-free in `debug` and `release` builds
/// (they return empty vectors or clamp where appropriate instead).
#[derive(Debug, Clone)]
pub struct LpcTool {
    pub order: usize,
    pub nbits: u8,
    pub max_coef: f64,
    pub min_coef: f64,
}

impl LpcTool {
    /* ---------- construction / helpers ---------- */

    pub fn new(order: usize, nbits: u8, max_coef: f64, min_coef: f64) -> Self {
        debug_assert!(nbits <= 31, "more than 31 bits is not useful here");
        LpcTool {
            order,
            nbits,
            max_coef,
            min_coef,
        }
    }

    fn safe_step(&self) -> f64 {
        let levels = (1_u32.checked_shl(self.nbits as u32).unwrap_or(1) - 1) as f64;
        // never allow division by zero
        (self.max_coef - self.min_coef).max(f64::EPSILON) / levels
    }

    /* ---------- uniform scalar quantiser ---------- */

    pub fn quantize_uniform(&self, value: f64) -> u32 {
        let clipped = value.clamp(self.min_coef, self.max_coef);
        let step = self.safe_step();
        ((clipped - self.min_coef) / step).round() as u32
    }

    pub fn dequantize_uniform(&self, index: u32) -> f64 {
        self.min_coef + (index as f64) * self.safe_step()
    }

    /* ---------- LPC core ---------- */

    /// Autocorrelation up to `order` lags.  
    /// If the requested order is larger than `x.len()-1` the tail is filled with zeros.
    fn autocorrelation(x: &[f64], order: usize) -> Vec<f64> {
        let n = x.len();
        let max_lag = order.min(n.saturating_sub(1));
        let mut r = vec![0.0; order + 1];

        for lag in 0..=max_lag {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += x[i] * x[i + lag];
            }
            r[lag] = sum;
        }
        r
    }

    /// Levinson–Durbin recursion (stable, regularised).
    fn levinson_durbin(r: &[f64], order: usize) -> Vec<f64> {
        let mut a = vec![0.0; order + 1];
        if r.is_empty() || r[0].abs() < 1e-12 {
            a[0] = 1.0;
            return a;
        }
        a[0] = 1.0;
        let mut e = r[0];

        for i in 1..=order {
            let mut acc = 0.0;
            for j in 1..i {
                acc += a[j] * r[i - j];
            }
            if e.abs() < 1e-12 {
                break;
            }
            let k = (r[i] - acc) / e;

            for j in 1..i {
                a[j] -= k * a[i - j];
            }
            a[i] = k;
            e *= 1.0 - k * k;
        }
        a
    }

    /// Compute standard LPC coefficients `a[0..=order]` from an **i32** signal.
    /// Returns a vector of length `order+1` (or empty if the input is empty).
    pub fn lpc(&self, signal: &[i32]) -> Vec<f64> {
        if signal.is_empty() {
            return Vec::new();
        }
        let signal_f: Vec<f64> = signal.iter().map(|&v| v as f64).collect();
        let r = Self::autocorrelation(&signal_f, self.order);
        let mut a = Self::levinson_durbin(&r, self.order);

        // Convert to the common representation where a[0] = 1 and a[1..] are negated.
        for coef in &mut a[1..] {
            *coef = -*coef;
        }
        a
    }

    /* ---------- prediction, encode/decode ---------- */

    fn predict(&self, signal: &[i32], a: &[f64]) -> Vec<i32> {
        let mut predicted = vec![0i32; signal.len()];
        let order = self.order.min(a.len().saturating_sub(1));

        for i in order..signal.len() {
            let mut sum_val = 0.0;
            for k in 1..=order {
                sum_val += a[k] * (signal[i - k] as f64);
            }
            predicted[i] = -(sum_val.round() as i32);
        }
        predicted
    }

    pub fn encode_lpc(&self, signal: &[i32], a: &[f64]) -> Vec<i32> {
        let predicted = self.predict(signal, a);
        signal
            .iter()
            .zip(predicted.iter())
            .map(|(orig, pred)| orig - pred)
            .collect()
    }

    pub fn decode_lpc(&self, residual: &[i32], a: &[f64]) -> Vec<i32> {
        let n = residual.len();
        let order = self.order.min(a.len().saturating_sub(1));
        let mut rec = vec![0i32; n];

        // seed with residual for the initial samples
        rec[..order.min(n)].copy_from_slice(&residual[..order.min(n)]);

        for i in order..n {
            let mut sum_val = 0.0;
            for k in 1..=order {
                sum_val += a[k] * (rec[i - k] as f64);
            }
            rec[i] = residual[i] - sum_val.round() as i32;
        }
        rec
    }

    /* ---------- convenience API ---------- */

    pub fn quantize_coefficients(&self, coefs: &[f64]) -> Vec<f64> {
        coefs
            .iter()
            .map(|&c| {
                let sym = self.quantize_uniform(c);
                self.dequantize_uniform(sym)
            })
            .collect()
    }

    /// Compute LPC, quantise coefficients (except `a[0]`), and return `(quantised a, residual)`.
    pub fn get_coefs_and_residuals(&self, signal: &[i32]) -> (Vec<f64>, Vec<i32>) {
        let mut a = self.lpc(signal);
        if a.is_empty() {
            return (a, Vec::new());
        }
        let quantised_tail = self.quantize_coefficients(&a[1..]);
        for (dst, src) in a[1..].iter_mut().zip(quantised_tail.iter()) {
            *dst = *src;
        }
        let residual = self.encode_lpc(signal, &a);
        (a, residual)
    }

    /// Map quantized symbol indices back to dequantized LPC coefficients.
    /// Indices are **clamped** to [0, levels] to be robust to corrupted inputs.
    pub fn coefs_from_symbols(&self, symbols: &[u32]) -> Vec<f64> {
        let levels = 1_u32
            .checked_shl(self.nbits as u32)
            .unwrap_or(1)
            .saturating_sub(1);
        symbols
            .iter()
            .map(|&sym| self.dequantize_uniform(sym.min(levels)))
            .collect()
    }

    /// Public, non-underscored counterpart of `_get_symbols`.
    pub fn symbols_from_coefs(&self, coefs: &[f64]) -> Vec<u32> {
        coefs.iter().map(|&c| self.quantize_uniform(c)).collect()
    }
}

/* -------------------------------------------------------------------------- */
/*                                   tests                                    */
/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::LpcTool;
    use rand::Rng;

    fn roundtrip(signal: &[i32], order: usize) {
        let tool = LpcTool::new(order, 8, 1.0, -1.0);
        let (a, res) = tool.get_coefs_and_residuals(signal);
        let reconstructed = tool.decode_lpc(&res, &a);
        assert_eq!(signal, &reconstructed[..]);
    }

    #[test]
    fn constant_signal() {
        let sig = vec![5; 32];
        roundtrip(&sig, 10);
    }

    #[test]
    fn impulse() {
        let mut sig = vec![0; 64];
        sig[0] = 1000;
        roundtrip(&sig, 12);
    }

    #[test]
    fn random_signal() {
        let mut rng = rand::rng();
        let sig: Vec<i32> = (0..256).map(|_| rng.random_range(-5000..=5000)).collect();
        roundtrip(&sig, 16);
    }

    #[test]
    fn short_signal_long_order() {
        // order is longer than len - 1
        let sig = vec![1, 2, 3];
        roundtrip(&sig, 20);
    }

    #[test]
    fn empty_signal() {
        let sig: Vec<i32> = Vec::new();
        let tool = LpcTool::new(10, 8, 1.0, -1.0);
        let (a, res) = tool.get_coefs_and_residuals(&sig);
        assert!(a.is_empty());
        assert!(res.is_empty());
    }

    #[test]
    fn quantiser_bounds() {
        let tool = LpcTool::new(1, 4, 0.5, -0.5);
        let idx_min = tool.quantize_uniform(-10.0);
        let idx_max = tool.quantize_uniform(10.0);
        assert_eq!(idx_min, 0);
        assert_eq!(idx_max, (1 << 4) - 1);
    }

    #[test]
    fn coefs_from_symbols_matches_quantize_coefficients() {
        let tool = LpcTool::new(10, 8, 1.0, -1.0);
        let coefs = vec![-1.0, -0.5, 0.0, 0.37, 0.99, 1.2];

        // coefs -> symbols -> coefs'  should match quantize_coefficients(coefs)
        let syms = tool.symbols_from_coefs(&coefs);
        let coefs_from_syms = tool.coefs_from_symbols(&syms);
        let coefs_q = tool.quantize_coefficients(&coefs);
        assert_eq!(coefs_from_syms, coefs_q);
    }

    #[test]
    fn coefs_from_symbols_clamps_out_of_range_indices() {
        let tool = LpcTool::new(5, 4, 0.5, -0.5);
        let levels = (1u32 << 4) - 1; // 15
        let symbols = vec![0, levels / 2, levels, levels + 1000, u32::MAX];
        let coefs = tool.coefs_from_symbols(&symbols);

        // First/last elements should be clamped to min/max coef
        assert!(*coefs.first().unwrap() >= tool.min_coef - f64::EPSILON);
        assert!(*coefs.last().unwrap() <= tool.max_coef + f64::EPSILON);
    }

    #[test]
    fn symbols_helpers_handle_empty_slices() {
        let tool = LpcTool::new(3, 8, 1.0, -1.0);
        let coefs: Vec<f64> = Vec::new();
        let symbols: Vec<u32> = Vec::new();
        assert!(tool.symbols_from_coefs(&coefs).is_empty());
        assert!(tool.coefs_from_symbols(&symbols).is_empty());
    }

}
