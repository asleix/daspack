// exp_golomb.rs

/// Convert signed to "unsigned" (zig-zag mapping):
///   if x > 0 -> 2*x - 1
///   if x <= 0 -> -2*x
#[inline]
fn signed_to_unsigned(x: i32) -> u64 {
    if x > 0 {
        let x = x as u64;
        (x << 1) - 1
    } else {
        let x = (-x) as u64;
        x << 1
    }
}

/// Convert "unsigned" back to signed. (inverse of above)
#[inline]
fn unsigned_to_signed(u: u64) -> i32 {
    if u % 2 == 0 {
        (-(u as i64) >> 1) as i32
    } else {
        (((u + 1) as i64) >> 1) as i32
    }
}

/* ---------------- Bit I/O (MSB-first) ---------------- */

struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    nbits: u8, // number of bits already filled in `cur`
}

impl BitWriter {
    fn new() -> Self {
        Self { out: Vec::new(), cur: 0, nbits: 0 }
    }

    #[inline]
    fn write_bit(&mut self, bit: u32) {
        // Place next bit at position (7 - nbits).
        self.cur |= ((bit & 1) as u8) << (7 - self.nbits);
        self.nbits += 1;
        if self.nbits == 8 {
            self.out.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    #[inline]
    fn write_bits(&mut self, value: u64, n: u32) {
        // MSB-first: emit from high to low.
        for i in (0..n).rev() {
            let b = ((value >> i) & 1) as u32;
            self.write_bit(b);
        }
    }

    #[inline]
    fn write_zeros(&mut self, n: u32) {
        for _ in 0..n {
            self.write_bit(0);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits != 0 {
            self.out.push(self.cur); // zero-padded to byte boundary
        }
        self.out
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_idx: usize,
    cur: u8,
    bits_left: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        if data.is_empty() {
            Self { data, byte_idx: 0, cur: 0, bits_left: 0 }
        } else {
            Self { data, byte_idx: 1, cur: data[0], bits_left: 8 }
        }
    }

    #[inline]
    fn read_bit(&mut self) -> Option<u8> {
        if self.bits_left == 0 {
            if self.byte_idx >= self.data.len() {
                return None;
            }
            self.cur = self.data[self.byte_idx];
            self.byte_idx += 1;
            self.bits_left = 8;
        }
        let b = (self.cur >> 7) & 1;
        self.cur <<= 1;
        self.bits_left -= 1;
        Some(b)
    }

    #[inline]
    fn read_bits(&mut self, n: u32) -> Option<u64> {
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | (self.read_bit()? as u64);
        }
        Some(v)
    }

    /// Read one order-0 Exp-Golomb value (EG0), returning the encoded *value* `v` (>= 1).
    /// Encoding rule: write `z = floor(log2(v))` zeros, then write `v` in `z+1` bits.
    fn read_eg0_value(&mut self) -> Option<u64> {
        // Count leading zeros until we hit a '1'.
        let mut zeros: u32 = 0;
        loop {
            let b = self.read_bit()?;
            if b == 0 {
                zeros += 1;
            } else {
                // We just consumed the leading '1' bit of v.
                let rest = if zeros > 0 { self.read_bits(zeros)? } else { 0 };
                let v = (1u64 << zeros) | rest;
                return Some(v);
            }
        }
    }
}

/* ---------------- EG_k encode/decode ---------------- */

/// Encode a slice of signed i32 using order-k Exponential-Golomb (MSB-first).
/// Returns a vector of bytes; output is zero-padded to byte alignment.
pub fn encode_k_expgolomb_list(data: &[i32], k: u32) -> Vec<u8> {
    let mut bw = BitWriter::new();

    for &val in data {
        let u = signed_to_unsigned(val);
        let q = u >> k;
        let r = if k > 0 { u & ((1u64 << k) - 1) } else { 0 };

        // EG0 for (q + 1):
        let v = q + 1; // v >= 1
        let z = 63u32.saturating_sub(v.leading_zeros() as u32); // floor(log2(v))
        // Number of leading zeros is z.
        bw.write_zeros(z);
        // Then write v in (z + 1) bits:
        bw.write_bits(v, z + 1);

        // Append remainder (k bits), MSB-first.
        if k > 0 {
            bw.write_bits(r, k);
        }
    }

    bw.finish()
}

/// Decode `count` values from k-Exponential-Golomb (MSB-first) byte stream.
pub fn decode_k_expgolomb_list(data: &[u8], count: usize, k: u32) -> Vec<i32> {
    let mut out = Vec::with_capacity(count);
    let mut br = BitReader::new(data);

    for _ in 0..count {
        // Read EG0 value v = q + 1
        let v = match br.read_eg0_value() {
            Some(v) => v,
            None => break, // truncated
        };
        let q = v - 1;

        // Read remainder (k bits)
        let r = if k > 0 {
            match br.read_bits(k) {
                Some(bits) => bits,
                None => break, // truncated
            }
        } else {
            0
        };

        let u = (q << k) | r;
        out.push(unsigned_to_signed(u));
    }

    out
}

/// A simple heuristic to estimate `k` from the mean of mapped unsigned values:
///   k = floor(log2(mean + 1))
pub fn estimate_best_k(data: &[i32]) -> u32 {
    if data.is_empty() {
        return 0;
    }
    let (sum, count) = data.iter().fold((0.0, 0usize), |(acc, cnt), &x| {
        (acc + signed_to_unsigned(x) as f64, cnt + 1)
    });
    let mean = (sum / count as f64).max(0.0);
    let k = (mean + 1.0).log2().floor();
    if k < 0.5 { 0 } else { k as u32 }
}

/* ---------------- Tests ---------------- */

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn seeded(seed: u64) -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    fn gen_vec_in_range(rng: &mut impl Rng, len: usize, low: i32, high: i32) -> Vec<i32> {
        (0..len).map(|_| rng.random_range(low..=high)).collect()
    }

    fn assert_roundtrip(data: &[i32], k: u32) {
        let enc = encode_k_expgolomb_list(data, k);
        let dec = decode_k_expgolomb_list(&enc, data.len(), k);
        assert_eq!(
            data, &dec[..],
            "roundtrip mismatch for k={k}, len={}, enc_len={}",
            data.len(),
            enc.len()
        );
    }

    #[test]
    fn signed_unsigned_inverse_smoke() {
        // Avoid i32::MIN (negation overflows with current mapping).
        let samples = [
            -10, -1, 0, 1, 2, 123, -123, 1_000_000, -1_000_000, i32::MAX, i32::MIN + 1,
        ];
        for &x in &samples {
            let u = super::signed_to_unsigned(x);
            let y = super::unsigned_to_signed(u);
            assert_eq!(x, y, "failed at x={x}, mapped u={u}");
        }
    }

    #[test]
    fn signed_unsigned_inverse_random() {
        let mut rng = seeded(0xC0FFEE);
        for _ in 0..10_000 {
            // Exclude i32::MIN to avoid overflow in signed_to_unsigned.
            let x: i32 = rng.random_range(i32::MIN + 1..=i32::MAX);
            let u = super::signed_to_unsigned(x);
            let y = super::unsigned_to_signed(u);
            assert_eq!(x, y);
        }
    }

    #[test]
    fn encode_decode_roundtrip_various_k_and_sizes() {
        let mut rng = seeded(0xA11CE);
        for &k in &[0u32, 1, 2, 3, 4, 5, 6, 8] {
            for &len in &[0usize, 1, 2, 3, 7, 16, 37, 128, 1000] {
                let data = gen_vec_in_range(&mut rng, len, -2_000, 2_000);
                assert_roundtrip(&data, k);
            }
        }
    }

    #[test]
    fn roundtrip_with_estimated_k_random() {
        let mut rng = seeded(0xDEC0DE);
        for _ in 0..20 {
            let len: usize = rng.random_range(0..=500);
            let data = gen_vec_in_range(&mut rng, len, -50, 50);
            let k = estimate_best_k(&data);
            let enc = encode_k_expgolomb_list(&data, k);
            let dec = decode_k_expgolomb_list(&enc, data.len(), k);
            assert_eq!(data, dec, "failed with estimated k={k}");
        }
    }

    #[test]
    fn decoder_ignores_extra_trailing_zero_padding() {
        let data = vec![0, 1, -1, 2, -2, 3, -3, 0, 0, 4];
        let k = 3;
        let mut enc = encode_k_expgolomb_list(&data, k);
        // Add extra all-zero bytes beyond the encoder's own padding.
        enc.extend_from_slice(&[0u8; 16]);

        let dec = decode_k_expgolomb_list(&enc, data.len(), k);
        assert_eq!(data, dec);
    }

    #[test]
    fn compact_with_k0_large_magnitudes() {
        // EG0 (k=0) should be compact even for moderately large magnitudes.
        let k = 0;
        let data = vec![10_000, -10_000, 0, 9_999, -9_999, 1, -1, 0];
        let enc = encode_k_expgolomb_list(&data, k);
        // This should be quite small (tens of bytes), not thousands.
        assert!(!enc.is_empty() && enc.len() < 64, "enc_len={}", enc.len());

        let dec = decode_k_expgolomb_list(&enc, data.len(), k);
        assert_eq!(data, dec);
    }

    #[test]
    fn all_negatives_roundtrip() {
        let mut data = Vec::with_capacity(256);
        for i in 0..256 {
            data.push(-i); // includes 0 at i=0; safely above i32::MIN
        }
        for &k in &[0u32, 1, 2, 4, 6] {
            assert_roundtrip(&data, k);
        }
    }

    #[test]
    fn extremes_roundtrip_with_sensible_k() {
        let data = vec![
            i32::MAX,
            i32::MIN + 1, // avoid MIN itself; current mapping would overflow
            -1,
            0,
            1,
            123_456_789,
            -123_456_789,
        ];
        for &k in &[12u32, 16, 20] {
            assert_roundtrip(&data, k);
        }
    }

    #[test]
    fn estimate_best_k_basics() {
        assert_eq!(estimate_best_k(&[]), 0);
        assert_eq!(estimate_best_k(&[0, 0, 0, 0]), 0);
        // Larger magnitudes should tend to increase k.
        let k_small = estimate_best_k(&[1, -1, 2, -2, 0, 0, 3]);
        let k_large = estimate_best_k(&[10_000, -10_000, 9_999, -9_999]);
        assert!(k_large >= k_small, "k_small={k_small}, k_large={k_large}");
        assert!(k_large >= 1);
    }
}
