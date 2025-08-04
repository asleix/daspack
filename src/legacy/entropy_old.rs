use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use constriction::stream::{
    queue::{DefaultRangeDecoder, DefaultRangeEncoder},
    Decode, Encode,
    model::DefaultLeakyQuantizer,
};
use numpy::ndarray::{Array2, Axis};
use std::io::Cursor;
use std::io::Read;

use crate::exp_golomb;


pub struct MuLawQuantizer {
    num_levels: i32,
    mu: f32,
    max_range: f32,
}

impl MuLawQuantizer {
    pub fn new(num_levels: i32, mu: f32, max_range: f32) -> Self {
        MuLawQuantizer {
            num_levels,
            mu,
            max_range,
        }
    }

    /// μ-law compression (input normalized to [-1, 1]).
    fn mu_law_compress(x: f32, mu: f32) -> f32 {
        x.signum() * (1.0 + mu * x.abs()).ln() / (1.0 + mu).ln()
    }

    /// μ-law expansion (output back to [-1, 1]).
    fn mu_law_expand(y: f32, mu: f32) -> f32 {
        y.signum()
            * ((1.0 + mu).powf(y.abs()) - 1.0)
            / mu
    }

    /// Quantize a single value (rough integer version).
    pub fn quantize_value(&self, x: f32) -> (i32, i32) {
        // Normalize to [-1, 1].
        let x_norm = x / self.max_range;
        let y = Self::mu_law_compress(x_norm, self.mu);
        // Map from [-1,1] to [0, num_levels-1].
        let idx_float = (y + 1.0) * (0.5 * (self.num_levels as f32 - 1.0));
        let mut idx = idx_float.round() as i32;
        idx = idx.clamp(0, self.num_levels - 1);

        // Map back to the quantized value in compressed domain
        let yq = (idx as f32) / (0.5 * (self.num_levels as f32 - 1.0)) - 1.0;
        let xq = Self::mu_law_expand(yq, self.mu) * self.max_range;

        (idx, xq.round() as i32)
    }

    /// Quantize an entire slice of values (returns quantized values and indices).
    pub fn quantize_slice(&self, xs: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let (indices, quants): (Vec<_>, Vec<_>) = xs.iter()
            .map(|&val| self.quantize_value(val as f32))
            .unzip();
        (quants, indices)
    }

    pub fn quantized_from_index(&self, idx: i32) -> i32 {
        let yq = (idx as f32) / (0.5 * (self.num_levels as f32 - 1.0)) - 1.0;
        let xq = Self::mu_law_expand(yq, self.mu) * self.max_range;
        xq.round() as i32
    }

    /// Recover quantized values from given indices.
    pub fn quantized_from_indices(&self, indices: &[i32]) -> Vec<i32> {
        indices
            .iter()
            .map(|&idx| {
                let yq = (idx as f32) / (0.5 * (self.num_levels as f32 - 1.0)) - 1.0;
                let xq = Self::mu_law_expand(yq, self.mu) * self.max_range;
                xq.round() as i32
            })
            .collect()
    }
}

/// Compute approximate percentile using quickselect.
fn percentile(arr: &[i32], pct: f64) -> i32 {
    if arr.is_empty() {
        return 0;
    }
    let mut cloned = arr.to_vec();
    let rank = ((pct / 100.0) * ((cloned.len() - 1) as f64)).round() as usize;
    let (_, elem, _) = cloned.select_nth_unstable(rank);
    *elem
}

/// Encodes the clipped row using a Laplace model.
/// Returns a tuple: (compressed_u32s, mu_index, scale_index, computed_mu, computed_scale).
#[inline]
fn encode_row_laplace(
    clipped: &[i32],
    param_quantizer: &MuLawQuantizer, 
    min_allowed: i32,
    max_allowed: i32,
) -> (Vec<u32>, i32, i32)
{
    // return (Vec::<u32>::new(), 0, 0);
    // Compute “mu” as the median:
    let mu_i32 = percentile(clipped, 50.0);
    // Mean absolute deviation around that median:
    let abs_dev = clipped
        .iter()
        .map(|&v| (v - mu_i32).abs() as f32)
        .sum::<f32>() / clipped.len() as f32;
    let scale_i32 = abs_dev.round() as i32;
    let scale_i32 = scale_i32.max(1);

    // Quantize parameters:
    let (mu_idx, mu_val) = param_quantizer.quantize_value(mu_i32 as f32);
    let (scale_idx, scale_val) = param_quantizer.quantize_value(scale_i32 as f32);

    // Range-encode using Laplace:
    let mut encoder = DefaultRangeEncoder::new();
    let quantizer = DefaultLeakyQuantizer::new(min_allowed..=max_allowed);

    let lap_model = quantizer.quantize(
        probability::distribution::Laplace::new(mu_val as f64, scale_val as f64)
    );

    encoder
        .encode_symbols(clipped.iter().map(|&sym| (sym, lap_model)))
        .unwrap();

    let compressed: Vec<u32> = encoder.into_compressed().unwrap();

    // Return everything needed to store or compare
    (compressed, mu_idx as i32, scale_idx as i32)
}

/// Encodes the clipped row using a Gaussian model.
/// Returns a tuple: (compressed_u32s, mu_index, scale_index, computed_mu, computed_scale).
#[inline]
fn encode_row_gaussian(
    clipped: &[i32],
    param_quantizer: &MuLawQuantizer, 
    min_allowed: i32,
    max_allowed: i32,
) -> (Vec<u32>, i32, i32)
{
    //return (Vec::<u32>::new(), 0, 0);
    // Compute “mu” as the mean:
    let mean = clipped.iter().copied().sum::<i32>() as f64 / clipped.len() as f64;

    // Compute standard deviation:
    let var = clipped.iter().map(|&v| {
        let dv = (v as f64) - mean;
        dv * dv
    }).sum::<f64>() / (clipped.len() as f64);

    let stdev = var.sqrt().round() as i32;
    let stdev = stdev.max(1);

    // Quantize parameters:
    let (mu_idx, mu_val) = param_quantizer.quantize_value(mean as f32);
    let (scale_idx, scale_val) = param_quantizer.quantize_value(stdev as f32);

    // Range-encode using Gaussian:
    let mut encoder = DefaultRangeEncoder::new();
    let quantizer = DefaultLeakyQuantizer::new(min_allowed..=max_allowed);

    let gauss_model = quantizer.quantize(
        probability::distribution::Gaussian::new(mu_val as f64, scale_val as f64)
    );

    encoder
        .encode_symbols(clipped.iter().map(|&sym| (sym, gauss_model)))
        .unwrap();

    let compressed: Vec<u32> = encoder.into_compressed().unwrap();
    (compressed, mu_idx as i32, scale_idx as i32)
}


/// Compress a 2D array of residuals in a manner similar to your Python code.
/// Returns a single 1D byte array that holds all encoded data.
/// Compress a 2D array of residuals.
pub fn compress_residuals(res: &Array2<i32>, tail_cut: f64) -> Vec<u8> {
    let mut buffer = Vec::<u8>::new();
    let height = res.nrows() as i32;
    let width = res.ncols() as i32;

    buffer.write_i32::<LittleEndian>(height).unwrap();
    buffer.write_i32::<LittleEndian>(width).unwrap();

    let mut rest_list = Vec::with_capacity((height * width) as usize);

    for h in 0..height as usize {
        let row = res.index_axis(Axis(0), h);
        let row_slice = row.as_slice().unwrap();

        let row_min_p = percentile(&row_slice, tail_cut);
        let row_max_p = percentile(&row_slice, 100.0 - tail_cut);

        let lim_quantizer = MuLawQuantizer::new(256, 255.0, 32767.0);

        let (row_min_idx, row_min_p ) = lim_quantizer.quantize_value(row_min_p as f32);
        let (row_max_idx, row_max_p ) = lim_quantizer.quantize_value(row_max_p as f32);

        let min_allowed = row_min_p - 1;
        let max_allowed = row_max_p + 1;

        let clipped: Vec<i32> = row_slice.iter()
            .map(|&x| x.clamp(min_allowed, max_allowed))
            .collect();

        rest_list.extend(
            row_slice.iter()
                .zip(&clipped)
                .map(|(&orig, &clip)| orig - clip.clamp(row_min_p, row_max_p))
        );

        let param_quantizer: MuLawQuantizer = MuLawQuantizer::new(256, 255.0, 16383.0);

        // 2) Try Laplace encoding:
        let (lap_comp, lap_mu_idx, lap_scale_idx) = 
            encode_row_laplace(&clipped, &param_quantizer, min_allowed, max_allowed);

        // 3) Try Gaussian encoding:
        let (gau_comp, gau_mu_idx, gau_scale_idx) =
            encode_row_gaussian(&clipped, &param_quantizer, min_allowed, max_allowed);

        let use_laplace = lap_comp.len() <= gau_comp.len();
        let dist_id = if use_laplace { 0u8 } else { 1u8 };

        buffer.write_u8(dist_id).unwrap();
        buffer.write_u8(row_min_idx as u8).unwrap();
        buffer.write_u8(row_max_idx as u8).unwrap();

        // 6) Write chosen mu/scale indices
        if use_laplace {
            buffer.write_u8(lap_mu_idx as u8).unwrap();
            buffer.write_u8(lap_scale_idx as u8).unwrap();
        } else {
            buffer.write_u8(gau_mu_idx as u8).unwrap();
            buffer.write_u8(gau_scale_idx as u8).unwrap();
        }

        // 7) Write the chosen compressed block
        let chosen_block = if use_laplace { &lap_comp } else { &gau_comp };
        let comp_len_bytes = (chosen_block.len() << 2) as i32;
        buffer.write_i32::<LittleEndian>(comp_len_bytes).unwrap();
        for &word in chosen_block {
            buffer.write_u32::<LittleEndian>(word).unwrap();
        }
    }

    let run_length_res: Vec<i32> = rest_list;
    let run_length_res_nonzero: Vec<i32> = run_length_res.into_iter().filter(|&x| x != 0).collect();
    let best_k = exp_golomb::estimate_best_k(&run_length_res_nonzero);

    buffer.write_u8(best_k as u8).unwrap();
    let encoded_rle = exp_golomb::encode_k_expgolomb_list(&run_length_res_nonzero, best_k);
    buffer.extend_from_slice(&encoded_rle);

    buffer
}

/// Decompress the data back into a 2D array of i32.
pub fn decompress_residuals(buffer: &[u8]) -> Array2<i32> {
    use std::io::Cursor;
    use byteorder::{LittleEndian, ReadBytesExt};

    // Create a cursor over the input buffer.
    let mut cursor = Cursor::new(buffer);

    // Read dimensions.
    let height = cursor.read_i32::<LittleEndian>().unwrap();
    let width = cursor.read_i32::<LittleEndian>().unwrap();
    let (height, width) = (height as usize, width as usize);

    // Preallocate the output array.
    let mut data = Array2::<i32>::zeros((height, width));

    // We'll collect correction indices (row, col) for later correction.
    let mut corrections = Vec::new();

    // Buffer to reuse for compressed data bulk reading.
    let mut comp_data_buf = Vec::<u32>::new();

    // Process each row.
    for h_idx in 0..height {
        // Read row parameters.
        let dist_id = cursor.read_u8().unwrap();
        let row_min_idx = cursor.read_u8().unwrap() as i32;
        let row_max_idx = cursor.read_u8().unwrap() as i32;
        let mu_idx = cursor.read_u8().unwrap() as i32;
        let scale_idx = cursor.read_u8().unwrap() as i32;

        let lim_quantizer = MuLawQuantizer::new(256, 255.0, 32767.0);
        let param_quantizer = MuLawQuantizer::new(256, 255.0, 16383.0);

        let row_min_p = lim_quantizer.quantized_from_index(row_min_idx);
        let row_max_p = lim_quantizer.quantized_from_index(row_max_idx);
        let mu_val = param_quantizer.quantized_from_index(mu_idx) as f64;
        let scale_val = param_quantizer.quantized_from_index(scale_idx).max(1) as f64;


        let min_allowed = row_min_p - 1;
        let max_allowed = row_max_p + 1;

        // Read the length of the compressed block for this row.
        let comp_len_bytes = cursor.read_i32::<LittleEndian>().unwrap() as usize;
        let comp_len_u32s = comp_len_bytes >> 2;

        // Bulk-read the compressed bytes directly from the buffer.
        let pos = cursor.position() as usize;
        let comp_bytes = &buffer[pos..pos + comp_len_bytes];
        cursor.set_position((pos + comp_len_bytes) as u64);

        // Convert the bytes slice to a u32 slice using align_to.
        // Convert the bytes slice to u32 values safely using chunking.
        assert_eq!(comp_bytes.len(), comp_len_u32s * 4);
        if comp_data_buf.len() < comp_len_u32s {
            comp_data_buf.resize(comp_len_u32s, 0);
        }
        for (i, chunk) in comp_bytes.chunks_exact(4).enumerate() {
            comp_data_buf[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        // Create the range decoder from the compressed data.
        let mut decoder =
            DefaultRangeDecoder::from_compressed(&comp_data_buf[..comp_len_u32s]).unwrap();

        // Set up the quantizer and Laplace model.
        let quantizer = DefaultLeakyQuantizer::new(min_allowed..=max_allowed);

        let decoded = if dist_id == 0 {
            let lap_model = quantizer.quantize(
                probability::distribution::Laplace::new(mu_val, scale_val)
            );
            decoder.decode_symbols((0..width).map(|_| lap_model))
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        } else {
            let gauss_model = quantizer.quantize(
                probability::distribution::Gaussian::new(mu_val, scale_val)
            );
           decoder
                .decode_symbols((0..width).map(|_| gauss_model))
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        
        let mut row = data.row_mut(h_idx);
        for (i, &val) in decoded.iter().enumerate() {
            row[i] = val;
            if val == min_allowed || val == max_allowed {
                corrections.push((h_idx, i));
            }
        }
        
    }

    // Read the remaining header for correction parameters.
    let best_k = cursor.read_u8().unwrap() as u32;
    let mut remaining = Vec::new();
    cursor.read_to_end(&mut remaining).unwrap();

    // Decode the exp-Golomb correction values.
    let exp_gol_values =
        exp_golomb::decode_k_expgolomb_list(&remaining, corrections.len(), best_k);

    // Apply corrections to the data array using the pre-collected correction indices.
    for (idx, (h_idx, col)) in corrections.into_iter().enumerate() {
        let val = exp_gol_values[idx];
        let sign_part = val.signum();
        data[[h_idx, col]] += val - sign_part;
    }

    data
}


#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn test_roundtrip() {
        // Make some dummy residuals
        let res = array![
            [10, 1000, -200, 10],
            [500, 501, -30000, 32000],
            [0, 3, 3, 5],
        ];

        let compressed = compress_residuals(&res, 5.0);
        let recovered = decompress_residuals(&compressed);

        assert_eq!(res.shape(), recovered.shape());
        // Check if the recovered is the same shape & close in value.
        // Depending on the mu-law quantization, might be off by 1 or so.
        // We'll do a looser check here.
        for ((r1, r2), idx) in res
            .iter()
            .zip(recovered.iter())
            .zip(0..)
        {
            let diff = (r1 - r2).abs();
            // With integer rounding, possibly small difference
            assert!(diff <= 1, "Mismatch at idx {}: {} vs {}", idx, r1, r2);
        }
    }
}
