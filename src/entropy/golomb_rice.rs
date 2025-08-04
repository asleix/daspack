//! Golomb–Rice coder for signed `i32` values.


// ===================== ZigZag mapping =====================

/// Map a signed 32-bit integer to an unsigned integer in a way that
/// places small magnitudes near zero (0 → 0, -1 → 1, 1 → 2, -2 → 3, ...).
/// The result fits in 32 bits; we widen to `u64` for convenient bit I/O.
#[inline]
pub fn zigzag_i32_to_u64(x: i32) -> u64 {
    // (x << 1) ^ (x >> 31) is the standard ZigZag transform for i32.
    ((x << 1) ^ (x >> 31)) as u32 as u64
}

/// Inverse ZigZag: recover the original `i32` from the mapped value.
/// Only the lower 32 bits are used; upper bits are ignored.
#[inline]
pub fn zigzag_u64_to_i32(u: u64) -> i32 {
    let n = u as u32;
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

// ===================== Bit writer (LSB-first) =====================

/// Accumulates bits LSB-first into a 64-bit register (`acc`), flushing
/// full bytes into `out` as they become available.
/// Invariant: the **lowest `nbits` bits** in `acc` are valid pending bits.
#[derive(Default)]
pub struct BitWriter {
    acc: u64,   // pending bits live in the low end
    nbits: u32, // number of valid pending bits in `acc` (0..=64)
    out: Vec<u8>,
}

impl BitWriter {
    /// Create an empty writer.
    #[allow(dead_code)]
    #[inline]
    pub fn new() -> Self {
        Self { acc: 0, nbits: 0, out: Vec::new() }
    }

    /// Create with a byte capacity hint (purely an optimization).
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self { acc: 0, nbits: 0, out: Vec::with_capacity(cap) }
    }

    /// Append `n` bits taken from the **LSBs** of `value` into the stream,
    /// keeping the stream LSB-first. Bits are appended *above* current `nbits`.
    #[inline]
    pub fn push_bits(&mut self, mut value: u64, mut n: u32) {
        while n > 0 {
            let space = 64 - self.nbits;
            let take = n.min(space);
            let mask = if take == 64 { u64::MAX } else { (1u64 << take) - 1 };

            // Place the next `take` bits right above the existing pending bits.
            self.acc |= (value & mask) << self.nbits;
            self.nbits += take;
            n -= take;

            // Discard the bits we just took from `value`.
            if take == 64 { value = 0; } else { value >>= take; }

            // Flush any newly completed bytes from the bottom (LSB).
            while self.nbits >= 8 {
                self.out.push(self.acc as u8);
                self.acc >>= 8;
                self.nbits -= 8;
            }
        }
    }

    /// Append a **unary** code with `q` zeros followed by a one.
    /// This is the quotient part of Golomb–Rice.
    #[inline]
    pub fn push_unary_q(&mut self, q: u64) {
        if q > 0 {
            self.push_bits(0, q as u32); // q zeros
        }
        self.push_bits(1, 1); // terminating one
    }

    /// Finish the stream: if a partial byte remains, emit it (implicitly
    /// zero-padding the high bits). Returns the owned byte buffer.
    #[inline]
    pub fn finalize(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

// ===================== Bit reader (LSB-first) =====================

/// Mirrors `BitWriter`: accumulates bytes into `acc` LSB-first and
/// returns bits from the low end. If input runs out, missing bits are 0.
/// Invariant: the **lowest `nbits` bits** in `acc` are valid unread bits.
pub struct BitReader<'a> {
    data: &'a [u8],
    idx: usize, // next byte to pull from `data`
    acc: u64,   // unread bits live in the low end
    nbits: u32, // number of valid unread bits (0..=64)
}

impl<'a> BitReader<'a> {
    /// Start reading from `data`.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, idx: 0, acc: 0, nbits: 0 }
    }

    /// Pull one byte from `data` and place it above current unread bits.
    /// Returns false if no more bytes are available.
    #[inline]
    fn refill_byte(&mut self) -> bool {
        if self.idx < self.data.len() {
            self.acc |= (self.data[self.idx] as u64) << self.nbits;
            self.nbits += 8;
            self.idx += 1;
            true
        } else {
            false
        }
    }

    /// Read `n` bits LSB-first. If the stream truncates, the remaining
    /// bits are read as zeros (useful for tolerant decoding of padding).
    #[inline]
    pub fn read_bits(&mut self, mut n: u32) -> u64 {
        let mut out = 0u64;
        let mut written = 0u32;

        while n > 0 {
            if self.nbits == 0 && !self.refill_byte() {
                break; // no more input: rest are zeros
            }
            let take = n.min(self.nbits);
            let mask = (1u64 << take) - 1;

            // Grab the next `take` bits from the low end.
            let chunk = self.acc & mask;
            out |= chunk << written;

            // Consume them from the accumulator.
            self.acc >>= take;
            self.nbits -= take;

            written += take;
            n -= take;
        }
        out
    }

    /// Read a unary run of zeros terminated by a one, returning `q`.
    /// If the stream truncates mid-run, returns the zeros seen so far.
    #[inline]
    pub fn read_unary_q(&mut self) -> u64 {
        let mut q: u64 = 0;
        loop {
            if self.nbits == 0 && !self.refill_byte() {
                return q; // truncated: never saw the terminating '1'
            }

            let tz = self.acc.trailing_zeros().min(self.nbits) as u64;
            q += tz;

            if tz == self.nbits as u64 {
                // All available bits were zeros: consume them and continue.
                self.acc >>= tz as u32;
                self.nbits = 0;
                continue; // refill on next iteration
            } else {
                // Found a '1' right after tz zeros: consume it too.
                let shift = (tz as u32) + 1;
                self.acc >>= shift;
                self.nbits -= shift;
                return q;
            }
        }
    }

}

// ===================== Golomb–Rice (k) : streaming API =====================

/// Encode one signed value using Rice parameter `k`.
/// Layout: **unary(q)** followed by **k-bit remainder** (LSB-first).
/// Here, `u = ZigZag(x)`, `q = u >> k`, `r = u & (2^k - 1)`.
#[inline]
pub fn rice_encode_one(bw: &mut BitWriter, x: i32, k: u32) {
    let k = k.min(32);                // mapped `u` fits in 32 bits for i32
    let u = zigzag_i32_to_u64(x);
    let q = if k == 0 { u } else { u >> k };
    let r = if k == 0 { 0 } else { u & ((1u64 << k) - 1) };

    bw.push_unary_q(q);
    if k > 0 {
        bw.push_bits(r, k);
    }
}

/// Decode one signed value using Rice parameter `k`.
/// Mirrors `rice_encode_one`. Tolerant of truncated padding.
#[inline]
pub fn rice_decode_one(br: &mut BitReader<'_>, k: u32) -> i32 {
    let k = k.min(32);
    let q = br.read_unary_q();
    let r = if k == 0 { 0 } else { br.read_bits(k) };
    let u = (q << k) | r;
    zigzag_u64_to_i32(u)
}

// ===================== Golomb–Rice (k) : buffer API =====================

/// Encode a slice of `i32` with Rice parameter `k` into a fresh `Vec<u8>`.
/// The output is byte-aligned: if the last byte is partial, high bits are zero.
pub fn rice_encode_list(data: &[i32], k: u32) -> Vec<u8> {
    // Small capacity hint to avoid repeated reallocations in common cases.
    let mut bw = BitWriter::with_capacity(data.len()  );
    for &x in data {
        rice_encode_one(&mut bw, x, k);
    }
    bw.finalize()
}

/// Decode exactly `count` `i32` values from a Rice-coded byte slice with parameter `k`.
/// Extra trailing zeros in `bytes` are ignored.
pub fn rice_decode_list(bytes: &[u8], count: usize, k: u32) -> Vec<i32> {
    let mut br = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(rice_decode_one(&mut br, k));
    }
    out
}

// ===================== Parameter selection =====================

/// Integer log2 for `x > 0`: returns floor(log2(x)).
#[inline]
fn ilog2_u64_nonzero(x: u64) -> u32 {
    debug_assert!(x > 0);
    63 - x.leading_zeros()
}

/// Fast heuristic for picking Rice `k` on a block of data:
/// Let `u = ZigZag(x)`. Minimizing an approximate expected length yields
///    2^k ≈ ln(2) * E[u]  ≈ 0.693 * mean(u).
/// This function uses a fixed-point integer approximation (693/1000).
pub fn estimate_best_k(data: &[i32]) -> u32 {
    if data.is_empty() {
        return 0;
    }

    // Accumulate mean of ZigZag-mapped magnitudes (saturating for safety).
    let mut sum: u64 = 0;
    for &x in data {
        sum = sum.saturating_add(zigzag_i32_to_u64(x));
    }
    let mean = sum / (data.len() as u64);
    if mean == 0 {
        return 0;
    }

    // Approximate ln(2) * mean with integer math; +clamp because k<=32 for i32.
    ilog2_u64_nonzero(mean).clamp(0, 32)
}


// ============================ Tests  ==============================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_roundtrip_edges() {
        let xs = [
            i32::MIN, i32::MIN + 1, -123456789, -1, 0, 1, 2, 123456789, i32::MAX
        ];
        for &x in &xs {
            let u = zigzag_i32_to_u64(x);
            let y = zigzag_u64_to_i32(u);
            assert_eq!(x, y, "x={x}, u={u}");
        }
    }

    #[test]
    fn unary_run_crosses_zero_byte() {
        let mut bw = BitWriter::new();
        bw.push_unary_q(8);          // 8 zeros, then a 1 → [0x00, 0x01] LSB-first
        let enc = bw.finalize();

        let mut br = BitReader::new(&enc);
        assert_eq!(br.read_unary_q(), 8);
    }


    #[test]
    fn rice_roundtrip_small_sets() {
        let data = [-3, -2, -1, 0, 1, 2, 3, 10, -10, 0, 0, 7, -7];
        for &k in &[0, 1, 2, 3, 4, 5] {
            let enc = rice_encode_list(&data, k);
            let dec = rice_decode_list(&enc, data.len(), k);
            assert_eq!(data.to_vec(), dec, "k={k}");
        }
    }

    #[test]
    fn rice_roundtrip_extremes() {
        let data = [i32::MIN, i32::MAX, -1, 0, 1, - (1 << 15), (1 << 15)];
        for &k in &[4, 8, 12, 16, 20, 24, 28, 32] {
            let enc = rice_encode_list(&data, k);
            let dec = rice_decode_list(&enc, data.len(), k);
            assert_eq!(data.to_vec(), dec, "k={k}");
        }
    }

    #[test]
    fn estimate_k_basic_behavior() {
        assert_eq!(estimate_best_k(&[]), 0);
        assert_eq!(estimate_best_k(&[0, 0, 0]), 0);
        let k_small = estimate_best_k(&[0, 1, -1, 2, -2, 0, 3, -3]);
        let k_large = estimate_best_k(&[10_000, -10_000, 9_999, -9_999]);
        assert!(k_large >= k_small, "k_small={k_small}, k_large={k_large}");
    }

    #[test]
    fn decoder_tolerates_trailing_zeros() {
        let data = [0, 1, -1, 2, -2, 3, -3, 0, 0, 4];
        let k = 3;
        let mut enc = rice_encode_list(&data, k);
        enc.extend_from_slice(&[0u8; 8]); // bogus padding
        let dec = rice_decode_list(&enc, data.len(), k);
        assert_eq!(data.to_vec(), dec);
    }
}
