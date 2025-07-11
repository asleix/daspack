use numpy::ndarray::{concatenate, s, Array2, Axis};
use rayon::prelude::*;
use std::cmp;

use crate::prediction;

/// A predictor for a 2D block using wavelet transforms and LPC.
/// In a production version, replace dummy operations with real implementations.

/// Processes 2D data in block‐wise fashion using the predictor.
pub struct BlockProcessor {
    block_height: usize,
    block_width: usize,
    lx: usize,
    lt: usize,
    lpc_order: usize,
}

impl BlockProcessor {
    pub fn new(
        block_height: usize,
        block_width: usize,
        lx: usize,
        lt: usize,
        lpc_order: usize,
    ) -> Self {
        Self {
            block_height,
            block_width,
            lx,
            lt,
            lpc_order,
        }
    }

    /// Encode the 2D data by splitting it into blocks and processing each block in parallel.
    ///
    /// Returns a tuple with:
    ///  - A vector of residual (transformed) blocks.
    ///  - A vector of tuples containing (row LPC coefficients, column LPC coefficients) for each block.
    pub fn encode(
        &self,
        data: &Array2<i32>,
    ) -> (Vec<Array2<i32>>, Vec<(Array2<f64>, Array2<f64>)>) {
        let (max_h, max_w) = data.dim();

        // Compute block starting coordinates.
        let blocks: Vec<(usize, usize)> = (0..max_h)
            .step_by(self.block_height)
            .flat_map(|h| (0..max_w).step_by(self.block_width).map(move |w| (h, w)))
            .collect();

        let lpc_tool = LpcTool::new(self.lpc_order, 6, 1.0, -1.0);
        let predictor = prediction::MultiBlockPredictor::new(self.lx, self.lt, lpc_tool);

        // Process each block concurrently.
        let results: Vec<_> = blocks
            .par_iter()
            .map(|&(h, w)| {
                let h_end = cmp::min(h + self.block_height, max_h);
                let w_end = cmp::min(w + self.block_width, max_w);

                // Extract the block from the data.
                let block = data.slice(s![h..h_end, w..w_end]).to_owned();

                // Process the block.
                let (residuals, row_coefs, col_coefs) = predictor.predict_diff(&block);

                // Optionally, remove the first LPC coefficient (assumed to be constant 1).
                if self.lpc_order > 0 {
                    let trimmed_row = row_coefs.slice(s![.., 1..]).to_owned();
                    let trimmed_col = col_coefs.slice(s![.., 1..]).to_owned();
                    (residuals, (trimmed_row, trimmed_col))
                } else {
                    (residuals, (row_coefs, col_coefs))
                }
            })
            .collect();

        // Unzip the results into separate vectors.
        let mut residuals_list = Vec::with_capacity(results.len());
        let mut lpc_coef_list = Vec::with_capacity(results.len());
        for (residuals, coefs) in results {
            residuals_list.push(residuals);
            lpc_coef_list.push(coefs);
        }
        (residuals_list, lpc_coef_list)
    }

    /// Decode the processed blocks to reconstruct the original data.
    ///
    /// Takes the list of residual blocks and LPC coefficients, and the shape of the original data.
    pub fn decode(
        &self,
        residuals_list: Vec<Array2<i32>>,
        lpc_coef_list: Vec<(Array2<f64>, Array2<f64>)>,
        data_shape: (usize, usize),
    ) -> Array2<i32> {
        let (max_h, max_w) = data_shape;
        let mut recon_data = Array2::<i32>::zeros((max_h, max_w));

        // Compute block starting coordinates.
        let blocks: Vec<(usize, usize)> = (0..max_h)
            .step_by(self.block_height)
            .flat_map(|h| (0..max_w).step_by(self.block_width).map(move |w| (h, w)))
            .collect();

        let lpc_tool = LpcTool::new(self.lpc_order, 6, 1.0, -1.0);
        let predictor = prediction::MultiBlockPredictor::new(self.lx, self.lt, lpc_tool);

        // Process each block concurrently to reconstruct.
        let block_recons: Vec<((usize, usize), Array2<i32>)> = blocks
            .into_par_iter()
            .enumerate()
            .map(|(i, (h, w))| {
                let h_end = cmp::min(h + self.block_height, max_h);
                let w_end = cmp::min(w + self.block_width, max_w);

                let (row_coefs, col_coefs) = &lpc_coef_list[i];

                // If LPC order > 0, prepend a column of ones (the constant coefficient).
                let full_row_coefs = if self.lpc_order > 0 {
                    let ones = Array2::from_elem((row_coefs.nrows(), 1), 1.0);
                    concatenate(numpy::ndarray::Axis(1), &[ones.view(), row_coefs.view()]).unwrap()
                } else {
                    row_coefs.clone()
                };

                let full_col_coefs = if self.lpc_order > 0 {
                    let ones = Array2::from_elem((col_coefs.nrows(), 1), 1.0);
                    concatenate(numpy::ndarray::Axis(1), &[ones.view(), col_coefs.view()]).unwrap()
                } else {
                    col_coefs.clone()
                };

                let residuals = residuals_list[i].clone();
                let block_recon =
                    predictor.reconstruct_diff(residuals, &full_row_coefs, &full_col_coefs);
                ((h, w), block_recon)
            })
            .collect();

        // Write each reconstructed block into the final data array.
        for ((h, w), block) in block_recons {
            let h_end = cmp::min(h + block.nrows(), max_h);
            let w_end = cmp::min(w + block.ncols(), max_w);

            // Copy the block slice into the appropriate position.
            recon_data
                .slice_mut(s![h..h_end, w..w_end])
                .assign(&block.slice(s![0..(h_end - h), 0..(w_end - w)]));
        }

        recon_data
    }
}


// Assume these are available from your project.
use crate::entropy::{compress_residuals, decompress_residuals};
use crate::prediction::{LpcTool, MultiBlockPredictor};

fn split_subbands_2d(res_block: &Array2<i32>, lx: usize, lt: usize) -> Vec<Array2<i32>> {
    if lx > 0 && lt > 0 {
        let (h, w) = res_block.dim();
        let h2 = h / 2;
        let w2 = w / 2;
        vec![
            res_block.slice(s![0..h2, 0..w2]).to_owned(),   // LL
            res_block.slice(s![0..h2, w2..w]).to_owned(),   // LH
            res_block.slice(s![h2..h, 0..w2]).to_owned(),   // HL
            res_block.slice(s![h2..h, w2..w]).to_owned(),   // HH
        ]
    } else {
        // If no wavelet transform, treat the entire residual as one chunk
        vec![res_block.to_owned()]
    }
}

fn combine_subbands_2d(subbands: &[Array2<i32>], lx: usize, lt: usize) -> Array2<i32> {
    if lx > 0 && lt > 0 {
        // Reconstruct the full residual by placing each sub‐band
        // in its correct position.
        let h2 = subbands[0].nrows();
        let w2 = subbands[0].ncols();
        let h = h2 * 2;
        let w = w2 * 2;

        let mut merged = Array2::<i32>::zeros((h, w));
        merged.slice_mut(s![0..h2, 0..w2]).assign(&subbands[0]);   // LL
        merged.slice_mut(s![0..h2, w2..w]).assign(&subbands[1]);   // LH
        merged.slice_mut(s![h2..h, 0..w2]).assign(&subbands[2]);   // HL
        merged.slice_mut(s![h2..h, w2..w]).assign(&subbands[3]);   // HH
        merged
    } else {
        // If no wavelet transform, there's only one sub‐band
        subbands[0].clone()
    }
}


/// A structure to hold the compressed representation of all blocks.
/// Instead of one long byte stream, we separate the data per block.
pub struct BlockCompressedData {
    /// Original data dimensions (height, width)
    pub data_shape: (usize, usize),
    /// For each block, the compressed residual bitstream.
    pub residuals: Vec<Vec<u8>>,
    /// For each block, the quantized LPC row coefficients (flattened).
    pub row_coefs: Vec<Vec<u8>>,
    /// For each block, the quantized LPC column coefficients (flattened).
    pub col_coefs: Vec<Vec<u8>>,
}

/// The compressor/decompressor that leverages block processing,
/// entropy coding for the residuals, and LPC quantization.
pub struct BlockCompress {
    block_height: usize,
    block_width: usize,
    lx: usize,
    lt: usize,
    lpc_order: usize,
    tail_cut: f64,
    block_processor: BlockProcessor,
    lpc_tool: LpcTool,
}

impl BlockCompress {
    /// Create a new BlockCompress instance with the given parameters.
    /// Uses default LPC tool parameters (8 bits, range [-2.0, 2.0]).
    pub fn new(
        block_height: usize,
        block_width: usize,
        lx: usize,
        lt: usize,
        lpc_order: usize,
        tail_cut: f64,
    ) -> Self {
        let block_processor = BlockProcessor::new(block_height, block_width, lx, lt, lpc_order);
        let lpc_tool = LpcTool::new(lpc_order, 8, 1.0, -1.0);
        Self {
            block_height,
            block_width,
            lx,
            lt,
            lpc_order,
            tail_cut,
            block_processor,
            lpc_tool,
        }
    }

    /// Compress the input 2D data.
    ///
    /// This method:
    ///  1. Splits the data into blocks and processes each block concurrently using
    ///     the BlockProcessor (to get residuals and LPC coefficients).
    ///  2. For each block, compresses the residuals with `compress_residuals`
    ///     (from crate::entropy).
    ///  3. Quantizes the LPC coefficients (row and column) using the LPC tool.
    ///
    /// The output is a `BlockCompressedData` structure that stores the compressed
    /// information per block.
    pub fn compress(&self, data: &Array2<i32>) -> BlockCompressedData {
        // First, split the data into blocks.
        let (residuals_list, lpc_coef_list) = self.block_processor.encode(data);
    
        // Process each block concurrently.
        let comp_results: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = residuals_list
            .into_par_iter()
            .zip(lpc_coef_list.into_par_iter())
            .map(|(res_block, (row_coefs, col_coefs))| {
                // Compress the residuals for this block.
                //let comp_res = crate::entropy::compress_residuals(&res_block, self.tail_cut);
                
                // Compress each sub‐band separately and concatenate.
                let mut comp_res_all = Vec::new();
                //Split the residual into sub‐bands if lx>0 && lt>0
                if self.lx > 0 && self.lt > 0 {
                    let subbands = split_subbands_2d(&res_block, self.lx, self.lt);
                    for sb in subbands {
                        let comp_sb = compress_residuals(&sb, self.tail_cut);
                        // Write sub‐band length as little‐endian u32, then bytes
                        let len_bytes = (comp_sb.len() as u32).to_le_bytes();
                        comp_res_all.extend_from_slice(&len_bytes);
                        comp_res_all.extend_from_slice(&comp_sb);
                    }
                } else {
                    let comp_res = crate::entropy::compress_residuals(&res_block, self.tail_cut);
                    comp_res_all.extend_from_slice(&comp_res);
                }

                // Quantize the LPC coefficients.
                let row_symbols: Vec<u8> = row_coefs
                    .iter()
                    .map(|&coef| self.lpc_tool.quantize_uniform(coef) as u8)
                    .collect();
                let col_symbols: Vec<u8> = col_coefs
                    .iter()
                    .map(|&coef| self.lpc_tool.quantize_uniform(coef) as u8)
                    .collect();
                (comp_res_all, row_symbols, col_symbols)
            })
            .collect();
    
        // Unzip the results.
        let mut comp_residuals = Vec::with_capacity(comp_results.len());
        let mut comp_row_coefs = Vec::with_capacity(comp_results.len());
        let mut comp_col_coefs = Vec::with_capacity(comp_results.len());
        for (res, row, col) in comp_results {
            comp_residuals.push(res);
            comp_row_coefs.push(row);
            comp_col_coefs.push(col);
        }
    
        BlockCompressedData {
            data_shape: data.dim(),
            residuals: comp_residuals,
            row_coefs: comp_row_coefs,
            col_coefs: comp_col_coefs,
        }
    }

    /// Decompress the compressed data to reconstruct the original 2D data.
    ///
    /// This method:
    ///  1. Computes the block coordinates from the original data shape.
    ///  2. For each block, decompresses the residuals (via `decompress_residuals`).
    ///  3. Dequantizes the LPC coefficients using the LPC tool and reconstructs the
    ///     full coefficient arrays (by prepending a column of ones if needed).
    ///  4. Reconstructs the block using `MultiBlockPredictor::reconstruct_diff`.
    ///  5. Inserts the reconstructed block into the final 2D array.
    pub fn decompress(&self, compressed: &BlockCompressedData) -> Array2<i32> {
        let (max_h, max_w) = compressed.data_shape;
        let mut recon_data = Array2::<i32>::zeros((max_h, max_w));
        
        // Compute block starting coordinates.
        let blocks: Vec<(usize, usize)> = (0..max_h)
            .step_by(self.block_height)
            .flat_map(|h| (0..max_w).step_by(self.block_width).map(move |w| (h, w)))
            .collect();
    
        // Reconstruct each block concurrently.
        let block_recons: Vec<((usize, usize), Array2<i32>)> = blocks
            .into_par_iter()
            .enumerate()
            .map(|(i, (h, w))| {
                let h_end = cmp::min(h + self.block_height, max_h);
                let w_end = cmp::min(w + self.block_width, max_w);
    
                // Decompress the residual block.
                //let res_block = crate::entropy::decompress_residuals(&compressed.residuals[i]);
                // --- Decompress the block's residual data, which may have sub‐bands. ---
                let mut raw_data = &compressed.residuals[i][..]; // slice
                let mut subbands = Vec::new();

                if self.lx > 0 && self.lt > 0 {
                    // Expect 4 sub‐bands
                    for _ in 0..4 {
                        let mut len_bytes = [0u8; 4];
                        len_bytes.copy_from_slice(&raw_data[0..4]);
                        raw_data = &raw_data[4..];
                        let sub_len = u32::from_le_bytes(len_bytes) as usize;
                        let sb_data = &raw_data[..sub_len];
                        raw_data = &raw_data[sub_len..];
                        // Decompress each sub‐band
                        subbands.push(decompress_residuals(sb_data));
                    }
                } else {
                    // Single sub‐band for no wavelets
                    subbands.push(decompress_residuals(raw_data));
                }

                // Combine sub‐bands back into a single residual block.
                let res_block = combine_subbands_2d(&subbands, self.lx, self.lt);

                // Dequantize row coefficients.
                let row_u32: Vec<u32> = compressed.row_coefs[i].iter().map(|&v| v as u32).collect();
                let row_coefs_vec: Vec<f64> = self.lpc_tool.from_symbols(&row_u32);
                let block_rows = h_end - h;
               
                let full_row_coefs = if self.lpc_order > 0 {
                    let row_coefs = Array2::from_shape_vec((block_rows, self.lpc_order), row_coefs_vec)
                    .expect("Invalid shape for row coefficients");
                    let ones = Array2::from_elem((block_rows, 1), 1.0);
                    concatenate(Axis(1), &[ones.view(), row_coefs.view()]).unwrap()
                } else {
                    Array2::<f64>::zeros((block_rows, 0))
                };
    
                // Dequantize column coefficients.
                let col_u32: Vec<u32> = compressed.col_coefs[i].iter().map(|&v| v as u32).collect();
                let col_coefs_vec: Vec<f64> = self.lpc_tool.from_symbols(&col_u32);
                let block_cols = w_end - w;

                let full_col_coefs = if self.lpc_order > 0 {
                    let col_coefs = Array2::from_shape_vec((block_cols, self.lpc_order), col_coefs_vec)
                    .expect("Invalid shape for column coefficients");
                    let ones = Array2::from_elem((block_cols, 1), 1.0);
                    concatenate(Axis(1), &[ones.view(), col_coefs.view()]).unwrap()
                } else {
                    Array2::<f64>::zeros((block_cols, 0))
                };
    
                // Reconstruct the block using the predictor.
                let predictor = crate::prediction::MultiBlockPredictor::new(self.lx, self.lt, self.lpc_tool.clone());
                let block_recon = predictor.reconstruct_diff(res_block, &full_row_coefs, &full_col_coefs);
    
                ((h, w), block_recon)
            })
            .collect();
    
        // Write each reconstructed block into the final image.
        for ((h, w), block) in block_recons {
            let h_end = cmp::min(h + block.nrows(), max_h);
            let w_end = cmp::min(w + block.ncols(), max_w);
            recon_data
                .slice_mut(s![h..h_end, w..w_end])
                .assign(&block.slice(s![0..(h_end - h), 0..(w_end - w)]));
        }
    
        recon_data
    }
}



// -----------------------
// Example usage:
//
// fn main() {
//     // Create dummy data.
//     let data = Array2::<i32>::from_elem((1024, 1024), 42);
//
//     // Create a block processor with desired parameters.
//     let processor = BlockProcessor::new(
//         128,  // block height
//         128,  // block width
//         2,    // lx: wavelet transform levels horizontally
//         2,    // lt: wavelet transform levels vertically
//         1,    // LPC order
//     );
//
//     // Encode the data in parallel.
//     let (residuals_list, lpc_coef_list) = processor.encode(&data);
//
//     // Decode to reconstruct the original data.
//     let recon_data = processor.decode(residuals_list, lpc_coef_list, data.dim());
//
//     // (Optionally, verify that `data` and `recon_data` are equal.)
// }
// -----------------------
