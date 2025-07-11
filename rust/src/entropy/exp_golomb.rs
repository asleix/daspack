//exp_golomb.rs

/// Convert signed to "unsigned" per your Python function:
///   if x > 0 -> 2*x - 1
///   if x <= 0 -> -2*x
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
fn unsigned_to_signed(u: u64) -> i32 {
    if u % 2 == 0 {
        (-(u as i64) >> 1) as i32
    } else {
        (((u + 1) as i64) >> 1) as i32
    }
}

/// Encode a slice of signed i32 using k‑exponential Golomb. 
/// Returns a vector of bits (as a `Vec<u8>`), appended with zero-padding
/// so that the result is byte-aligned.
pub fn encode_k_expgolomb_list(data: &[i32], k: u32) -> Vec<u8> {
    // We'll build bits in a u64 buffer and flush them to bytes.
    let mut out_bits = Vec::<u8>::new();
    let mut bit_buffer: u64 = 0;
    let mut bits_in_buffer = 0;

    // A small lambda for writing bits to `out_bits`.
    let mut push_bits = |mut value: u64, num_bits: usize| {
        // Add bits from the bottom of 'value' to the bit_buffer
        // (lowest bit first). We'll keep them left to right for convenience.
        let mut left = num_bits;
        while left > 0 {
            let take = left.min(64 - bits_in_buffer);
            // Shift existing bits up, then OR in the new bits
            let mask = if take == 64 {
                u64::MAX // If take == 64, (1u64 << 64) is invalid.
            } else {
                (1u64 << take) - 1
            };
        
            bit_buffer |= (value & mask) << bits_in_buffer;
            bits_in_buffer += take;
            value >>= take.min(31);
            left -= take;
            // If buffer is full, flush to bytes
            while bits_in_buffer >= 8 {
                out_bits.push(bit_buffer as u8);
                bit_buffer >>= 8;
                bits_in_buffer -= 8;
            }
        }
    };

    for &val in data {
        let u_val = signed_to_unsigned(val);
        let q = u_val >> k; // quotient
        let remainder = u_val & ((1 << k) - 1);

        // Encode unary: q zeros followed by a 1
        push_bits(0, q as usize);
        push_bits(1, 1); 

        // push remainder, which is k bits
        if k > 0 {
            push_bits(remainder, k as usize);
        }
    }

    // Byte align the final partial bits if needed:
    if bits_in_buffer > 0 {
        while bits_in_buffer >= 8 {
            // flush full bytes if any
            out_bits.push(bit_buffer as u8);
            bit_buffer >>= 8;
            bits_in_buffer -= 8;
        }
        if bits_in_buffer > 0 {
            out_bits.push(bit_buffer as u8);
        }
    }

    out_bits
}

/// Decode a slice of k‑exponential Golomb-coded bits, returning exactly `count` i32 values.
pub fn decode_k_expgolomb_list(data: &[u8], count: usize, k: u32) -> Vec<i32> {
    let mut out = Vec::with_capacity(count);
    let mut bit_buffer: u64 = 0;
    let mut bits_in_buffer = 0;
    let mut byte_idx = 0;

    for _ in 0..count {
        // Decode unary code to find q
        let mut q = 0u32;
        loop {
            if bits_in_buffer == 0 {
                if byte_idx >= data.len() {
                    break; // Truncated input
                }
                bit_buffer = data[byte_idx] as u64;
                byte_idx += 1;
                bits_in_buffer = 8;
            }

            let valid_bits = bits_in_buffer;
            let tz = (bit_buffer.trailing_zeros()).min(valid_bits);

            if tz == valid_bits {
                // All bits are zeros
                q += valid_bits;
                bits_in_buffer = 0;
            } else {
                // Found a 1 after tz zeros
                q += tz;
                bits_in_buffer -= tz + 1;
                bit_buffer >>= tz + 1;
                break;
            }
        }

        // Decode remainder (k bits)
        let mut remainder = 0u32;
        let mut remaining = k;
        while remaining > 0 {
            if bits_in_buffer == 0 {
                if byte_idx >= data.len() {
                    break;
                }
                bit_buffer = data[byte_idx] as u64;
                byte_idx += 1;
                bits_in_buffer = 8;
            }

            let take = remaining.min(bits_in_buffer);
            let mask = (1 << take) - 1;
            let chunk = (bit_buffer & mask) as u32;
            remainder |= chunk << (k - remaining);
            bit_buffer >>= take;
            bits_in_buffer -= take;
            remaining -= take;
        }

        let u_val = ((q as u64) << (k as u64)) + remainder as u64;
        out.push(unsigned_to_signed(u_val));
    }

    out
}


/// A simple function to estimate `k`
///   k = floor(log2(mean_val + 1))
/// for the mapped unsigned data.
pub fn estimate_best_k(data: &[i32]) -> u32 {
    if data.is_empty() {
        return 0;
    }

    // Accumulate sum and count
    let (sum, count) = data.iter().fold((0.0, 0), |(acc, cnt), &x| {
        let ux = signed_to_unsigned(x) as f64;
        (acc + ux, cnt + 1)
    });

    let mean_val = (sum / (count as f64)).max(0.0);
    let k = (mean_val + 1.0).log2().floor();

    if k < 0.5 {
        0
    } else {
        k as u32
    }
}