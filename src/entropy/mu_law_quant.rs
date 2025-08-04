//mu_law_quant.rs

// ──────────────────────────────────────────────────────────────────────────
//  µ‑law scalar quantiser (256 levels)
// ──────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct MuLawQuantizer {
    mu: f32,
    max: f32,
    lut: Box<[i32; 256]>, // de‑quant table
}

impl MuLawQuantizer {
    pub fn new(mu: f32, max: f32) -> Self {
        let mut lut = [0i32; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            let y = (i as f32) / 255.0 * 2.0 - 1.0; // in [‑1,1]
            *v = (Self::mu_law_expand(y, mu) * max).round() as i32;
        }
        Self { mu, max, lut: Box::new(lut) }
    }

    #[inline] fn mu_law_compress(x: f32, mu: f32) -> f32 {
        x.signum() * ((1.0 + mu * x.abs()).ln() / (1.0 + mu).ln())
    }
    #[inline] fn mu_law_expand(y: f32, mu: f32) -> f32 {
        y.signum() * ((1.0 + mu).powf(y.abs()) - 1.0) / mu
    }

    #[inline]
    pub fn quantize(&self, x: f32) -> (u8, i32) {
        let norm = (x / self.max).clamp(-1.0, 1.0);
        let y = Self::mu_law_compress(norm, self.mu);
        let idx = ((y + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
        (idx, self.lut[idx as usize])
    }

    #[inline]
    pub fn quantize_ceil(&self, x: f32) -> (u8, i32) {
        let (idx, val) = self.quantize(x);
        if val as f32 >= x { return (idx, val); }
        let idx_up = idx.saturating_add(1);
        (idx_up, self.lut[idx_up as usize])
    }

    #[inline] pub fn dequantize(&self, idx: u8) -> i32 { self.lut[idx as usize] }
}
