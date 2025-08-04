// fast_models.rs

use constriction::stream::model::{EntropyModel, DecoderModel, EncoderModel};
use std::num::NonZeroU16;
use crate::entropy::ans_tables::{PREC, MAX_VAL, nz, LAPLACE_CDF, GAUSSIAN_CDF, TABLE_LEN};

/* -------- generic helper macro to avoid repetition ---------------- */
macro_rules! make_model {
    ($name:ident, $tbl:ident) => {
        #[derive(Clone, Copy)]
        pub struct $name { pub sc_idx: u8, pub mu: i32 }

        /* ---------- marker ---------------------------------------- */
        impl EntropyModel<PREC> for $name {
            type Symbol      = i32;
            type Probability = u16;
        }

        /* ---------- decoder path ---------------------------------- */
        impl DecoderModel<PREC> for $name {
            #[inline(always)]
            fn quantile_function(
                &self,
                q: u16,
            ) -> (i32, u16, NonZeroU16) {
                let cdf = &$tbl[self.sc_idx as usize];
                // binary search over the row (12 iterations max)
                let mut lo = 0usize;
                let mut hi = cdf.len() - 1;
                while lo + 1 < hi {
                    let mid = (lo + hi) >> 1;
                    if q < cdf[mid] { hi = mid; } else { lo = mid; }
                }
                let sym  = lo as i32 - MAX_VAL + self.mu;
                let left = cdf[lo];
                (sym, left, nz(cdf[lo + 1] - left))
            }
        }

        /* ---------- optional encoder path (for exact symmetry) ---- */
        impl EncoderModel<PREC> for $name {
            #[inline(always)]
            fn left_cumulative_and_probability(
                &self,
                symbol: impl core::borrow::Borrow<Self::Symbol>,
            ) -> Option<(u16, NonZeroU16)> {
                let idx = *symbol.borrow() - self.mu + MAX_VAL;
                if (0..TABLE_LEN as i32).contains(&idx) {
                    let i = idx as usize + 1;               // +1 because cdf[0] = 0
                    let cdf = &$tbl[self.sc_idx as usize];
                    let prob = cdf[i] - cdf[i - 1];
                    Some((cdf[i - 1], nz(prob)))
                } else {
                    None
                }
            }
        }
    };
}

/* ---- concrete instantiations ------------------------------------ */
make_model!(FastLaplace , LAPLACE_CDF);
make_model!(FastGaussian, GAUSSIAN_CDF);

