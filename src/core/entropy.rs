// Fast residual compressor for DAS data (full source)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::{Array2, Axis};
use std::io::{Cursor, Read};

pub mod golomb_rice;
// mod fast_models;
// mod mu_law_quant;
// mod ans_tables;
// pub mod coders;



/* ────────────────────────────────────────────────────────────────────
 * Error handling
 * ────────────────────────────────────────────────────────────────── */

#[derive(thiserror::Error, Debug)]
pub enum CodecError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Corrupted bit-stream: {0}")]
    Corrupted(&'static str),
    #[error("Row too large: {0} > 2^22 Bytes")]
    RowTooLarge(usize),
    #[error("Wrong Exp-Gol encoding shape")]
    ExpGolShape(#[from] ndarray::ShapeError),
}


type Result<T, E = CodecError> = std::result::Result<T, E>;


#[inline]
fn write_raw_payload(bytes: &[u8], dst: &mut Vec<u8>) -> Result<()> {
    if bytes.len() > 0xFF_FFFF {
        return Err(CodecError::RowTooLarge(bytes.len()));
    }
    let len24 = (bytes.len() as u32).to_le_bytes();
    dst.extend_from_slice(&len24[..3]);
    dst.extend_from_slice(bytes);
    Ok(())
}
#[inline]
fn read_len24(cur: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut b = [0u8; 4];
    cur.read_exact(&mut b[..3])?;
    Ok(u32::from_le_bytes(b))
}
#[inline]
fn take_slice<'a>(
    buf: &'a [u8],
    cur: &mut Cursor<&[u8]>,
    bytes: usize,
) -> Result<&'a [u8]> {
    let start = cur.position() as usize;
    let end = start + bytes;
    if end > buf.len() {
        return Err(CodecError::Corrupted("payload overrun"));
    }
    cur.set_position(end as u64);
    Ok(&buf[start..end])
}



/* ────────────────────────────────────────────────────────────────────
 * Rice-only compressor (row-wise)
 * ────────────────────────────────────────────────────────────────── */

/// Compress residuals row-wise using Golomb–Rice coding only.
/// Format:
///   i32 h, i32 w,
///   for each row r in 0..h:
///       u8 k_r,
///       u24 len_r,
///       [len_r bytes of Rice payload for exactly w values]
pub fn compress_residuals_rice(res: ndarray::ArrayView2<'_, i32>) -> Result<Vec<u8>> {
    let (h, w) = res.dim();

    // Capacity hint: Rice is often compact; start with ~half of raw size.
    let mut out = Vec::<u8>::with_capacity(h * w / 2);

    /* ---- header ------------------------------------------------ */
    out.write_i32::<LittleEndian>(h as i32)?;
    out.write_i32::<LittleEndian>(w as i32)?;

    /* ---- per-row payloads -------------------------------------- */
    for row in res.axis_iter(Axis(0)) {
        let xs = row.as_slice().expect("row view must be contiguous (C-order)");

        // Pick k per row and encode.
        let mut k = golomb_rice::estimate_best_k(xs);

        // Safety check: cap the maximum quotient q = (u >> k).
        const QCAP: u64 = 4096; 
        let mut max_u: u64 = 0;
        for &x in xs {
            let u = golomb_rice::zigzag_i32_to_u64(x);
            if u > max_u {
                max_u = u;
            }
        }
        while k < 32 && (max_u >> k) > QCAP {
            k += 1;
        }

        let bytes = golomb_rice::rice_encode_list(xs, k);

        // Write: k (u8) + 24-bit length + payload
        out.write_u8(k as u8)?;
        write_raw_payload(&bytes, &mut out)?; // reuses existing helper
    }

    Ok(out)
}

/* ────────────────────────────────────────────────────────────────────
 * Rice-only decompressor (row-wise)
 * ────────────────────────────────────────────────────────────────── */

/// Decompress residuals encoded by `compress_residuals_rice`.
pub fn decompress_residuals_rice(buf: &[u8]) -> Result<Array2<i32>> {
    let mut cur = Cursor::new(buf);

    /* ---- header ------------------------------------------------ */
    let h = cur.read_i32::<LittleEndian>()? as usize;
    let w = cur.read_i32::<LittleEndian>()? as usize;

    let mut out = Array2::<i32>::zeros((h, w));

    /* ---- per-row payloads -------------------------------------- */
    for r in 0..h {
        // Read k and payload length.
        let k = cur.read_u8()? as u32;
        let bytes = read_len24(&mut cur)? as usize;

        // Slice into the buffer and decode exactly `w` values.
        let slice = take_slice(buf, &mut cur, bytes)?;
        let vals = golomb_rice::rice_decode_list(slice, w, k);

        // Copy into the row.
        out.row_mut(r)
            .as_slice_mut()
            .expect("row not contiguous")
            .copy_from_slice(&vals);
    }

    Ok(out)
}

/* ────────────────────────────────────────────────────────────────────
 * Tests
 * ────────────────────────────────────────────────────────────────── */
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn roundtrip_fixed() {
        let data = array![
            [10, 1000, -200, 10],
            [500, 501, -30000, 32000],
            [0, 0, 0, 0],
        ];
        let bytes = compress_residuals_rice(data.view()).unwrap();
        let recon = decompress_residuals_rice(&bytes).unwrap();
        assert_eq!(data, recon);
    }

    #[test]
    fn roundtrip_random() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0xDADBEEF);
        let res: Array2<i32> = Array2::from_shape_fn((128, 64), |_| rng.random_range(-32768..32768));
        let bytes = compress_residuals_rice(res.view()).unwrap();
        let recon = decompress_residuals_rice(&bytes).unwrap();
        assert_eq!(res, recon);
    }
}
