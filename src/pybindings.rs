
use numpy::ndarray::{Array1, Array2};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

use crate::wavelets;
use crate::prediction;
use crate::blocks;
use crate::entropy::exp_golomb;
use crate::entropy;


/// ---------- Python bindings for calling internal functions ------------
// #[allow(non_snake_case)]
#[pymodule]
fn compute<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    /// Wavelet Transforms
    #[pyfn(m)]
    #[pyo3(name = "fwd_dwt_txfm_1d")]
    fn fwd_dwt_txfm_1d<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, i32>,
    ) -> Bound<'py, PyArray1<i32>> {

        // 2) Copy to a Rust-owned array
        let mut local: Array1<i32> =  x.as_array().to_owned();

        // 3) Perform the wavelet transform on the local copy
        wavelets::fwd_txfm_1d_inplace(local.view_mut());
        
        local.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "inv_dwt_txfm_1d")]
    fn inv_dwt_txfm_1d<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, i32>,
    ) -> Bound<'py, PyArray1<i32>> {

        // 2) Copy to a Rust-owned array
        let mut local: Array1<i32> =  x.as_array().to_owned();

        // 3) Perform the wavelet transform on the local copy
        wavelets::inv_txfm_1d_inplace(local.view_mut());
        
        local.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "fwd_dwt_txfm_2d")]
    fn fwd_dwt_txfm_2d<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, i32>,
    ) -> Bound<'py, PyArray2<i32>> {

        // 2) Copy to a Rust-owned array
        let mut local: Array2<i32> =  x.as_array().to_owned();

        // 3) Perform the wavelet transform on the local copy
        wavelets::fwd_txfm2d_inplace(&mut local);
        
        local.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "inv_dwt_txfm_2d")]
    fn inv_dwt_txfm_2d<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, i32>,
    ) -> Bound<'py, PyArray2<i32>> {

        // 2) Copy to a Rust-owned array
        let mut local: Array2<i32> =  x.as_array().to_owned();

        // 3) Perform the wavelet transform on the local copy
        wavelets::inv_txfm2d_inplace(&mut local);
        
        local.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "fwd_dwt_txfm_2d_levels")]
    fn fwd_dwt_txfm_2d_levels<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, i32>,
        levels: usize,
    ) -> Bound<'py, PyArray2<i32>> {

        // 2) Copy to a Rust-owned array
        let mut local: Array2<i32> =  x.as_array().to_owned();

        // 3) Perform the wavelet transform on the local copy
        wavelets::fwd_txfm2d_levels_inplace(&mut local, levels);
        
        local.to_pyarray(py)
    }


    #[pyfn(m)]
    #[pyo3(name = "inv_dwt_txfm_2d_levels")]
    fn inv_dwt_txfm_2d_levels<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, i32>,
        levels: usize,
    ) -> Bound<'py, PyArray2<i32>> {

        // 2) Copy to a Rust-owned array
        let mut local: Array2<i32> =  x.as_array().to_owned();

        // 3) Perform the wavelet transform on the local copy
        wavelets::inv_txfm2d_levels_inplace(&mut local, levels);
        
        local.to_pyarray(py)
    }


    /// Block Prediction
    #[pyfn(m)]
    #[pyo3(name = "predict_block")]
    #[allow(clippy::type_complexity)]
    fn predict_block<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, i32>,
        levels: usize,
        lpc_order: usize,
    ) -> (Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)  {

        let local: Array2<i32> =  x.as_array().to_owned();
        let lpc_tool =  prediction::LpcTool::new(lpc_order, 6, 1.0, -1.0);
        let predictor = prediction::MultiBlockPredictor::new(levels, levels, lpc_tool);

        let (resid, row_coefs, col_coefs) = predictor.predict_diff(&local);

        (
            resid.to_pyarray(py),
            row_coefs.to_pyarray(py),
            col_coefs.to_pyarray(py),
        )

    }

    #[pyfn(m)]
    #[pyo3(name = "reconstruct_block")]
    fn reconstruct_block<'py>(
        py: Python<'py>,
        resid: PyReadonlyArray2<'py, i32>,
        row_coefs: PyReadonlyArray2<'py, f64>,
        col_coefs: PyReadonlyArray2<'py, f64>,
        levels: usize,
        lpc_order: usize,
    ) -> Bound<'py, PyArray2<i32>> {

        let local_x: Array2<i32> =  resid.as_array().to_owned();
        let local_r: Array2<f64> =  row_coefs.as_array().to_owned();
        let local_c: Array2<f64> =  col_coefs.as_array().to_owned();

        let lpc_tool =  prediction::LpcTool::new(lpc_order, 6, 1.0, -1.0);
        let predictor = prediction::MultiBlockPredictor::new(levels, levels, lpc_tool);

        let recon = predictor.reconstruct_diff(local_x, &local_r, &local_c);

        recon.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "compress_data")]
    fn compress_data<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<'py, i32>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let local_data: Array2<i32> =  data.as_array().to_owned();
        
        let p = blocks::CompressParams::new(local_data.shape()[0], local_data.shape()[1], 1, 1, 1);

        let bitstream = blocks::compress_lossless(&local_data, p).expect("compress_lossless failed");
        
        Ok(bitstream.to_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "decompress_data")]
    fn decompress_data<'py>(
        py: Python<'py>,
        bitstream: PyReadonlyArray1<'py, u8>,
        shape: (usize, usize)
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {

        let local_bytes= bitstream.to_vec()?;
        let p = blocks::CompressParams::new(shape.0, shape.1, 1, 1, 1);

        let data = blocks::decompress_lossless(&local_bytes, shape, p).expect("decompress failed");
        Ok(data.to_pyarray(py))
    }



/*    
    // Block-based processing
    #[pyfn(m)]
    #[pyo3(name = "compress_block")]
    fn compress_block<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<'py, i32>,
        block_height: usize,
        block_width: usize,
        lx: usize,
        lt: usize,
        lpc_order: usize,
        tail_cut: f64,
    ) -> (Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>) {
        // Get the underlying array view (no need to clone if compress() accepts a view)

        let local_data: Array2<i32> =  data.as_array().to_owned();


        let compressor = blocks::BlockCompress::new(block_height, block_width, lx, lt, lpc_order, tail_cut);
        let cmp_data = compressor.compress(&local_data);
    
        // Convert each Vec<u8> into a PyArray1 and collect into a PyList
        let py_list_residuals = PyList::new(
            py,
            cmp_data
                .residuals
                .into_iter()
                .map(|row| PyArray1::from_vec(py, row))
                .collect::<Vec<_>>(),
        );
    
        let py_list_row_coefs = PyList::new(
            py,
            cmp_data
                .row_coefs
                .into_iter()
                .map(|row| PyArray1::from_vec(py, row))
                .collect::<Vec<_>>(),
        );
    
        let py_list_col_coefs = PyList::new(
            py,
            cmp_data
                .col_coefs
                .into_iter()
                .map(|row| PyArray1::from_vec(py, row))
                .collect::<Vec<_>>(),
        );
    
        (
            py_list_residuals.unwrap(),
            py_list_row_coefs.unwrap(),
            py_list_col_coefs.unwrap(),
        )
    }
    
    #[pyfn(m)]
    #[pyo3(name = "decompress_block")]
    fn decompress_block<'py>(
        py: Python<'py>,
        py_list_residuals: &Bound<'py, PyList>,
        py_list_row_coefs: &Bound<'py, PyList>,
        py_list_col_coefs: &Bound<'py, PyList>,
        data_height: usize,
        data_width: usize,
        block_height: usize,
        block_width: usize,
        lx: usize,
        lt: usize,
        lpc_order: usize,
        tail_cut: f64,
    ) -> Bound<'py, PyArray2<i32>> {
        let compressor = blocks::BlockCompress::new(block_height, block_width, lx, lt, lpc_order, tail_cut);
    
        // Convert each PyList item (expected to be a 1D NumPy array of u8) into a Vec<u8>
        let comp_residuals: Vec<Vec<u8>> = py_list_residuals
            .iter()
            .map(|item| {
                let array = item.downcast::<PyArray1<u8>>().expect("Expected a PyArray1<u8>");
                array.readonly().as_slice().expect("Could not get slice").to_vec()
            })
            .collect();
    
        let comp_row_coefs: Vec<Vec<u8>> = py_list_row_coefs
            .iter()
            .map(|item| {
                let array = item.downcast::<PyArray1<u8>>().expect("Expected a PyArray1<u8>");

                array.readonly().as_slice().expect("Could not get slice").to_vec()
            })
            .collect();
    
        let comp_col_coefs: Vec<Vec<u8>> = py_list_col_coefs
            .iter()
            .map(|item| {
                let array = item.downcast::<PyArray1<u8>>().expect("Expected a PyArray1<u8>");

                array.readonly().as_slice().expect("Could not get slice").to_vec()
            })
            .collect();
    
        // Construct the compressed data structure
        let blck_cmp_data = blocks::BlockCompressedData {
            data_shape: (data_height, data_width),
            residuals: comp_residuals,
            row_coefs: comp_row_coefs,
            col_coefs: comp_col_coefs,
        };
    
        let recon_data: Array2<i32> = compressor.decompress(&blck_cmp_data);
        recon_data.to_pyarray(py)
    }
     */
    

    // Exp-golomb

    #[pyfn(m)]
    #[pyo3(name = "encode_exp_golomb")]
    fn encode_exp_golomb<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, i32>,
        k: u32,
    ) -> Bound<'py, PyArray1<u8>> {

        let local =  x.as_array().to_owned().to_vec();

        let comp = exp_golomb::encode_k_expgolomb_list(&local, k);

        PyArray1::from_vec(py, comp).to_owned()
    }

    #[pyfn(m)]
    #[pyo3(name = "decode_exp_golomb")]
    fn decode_exp_golomb<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, u8>,
        count: usize,
        k: u32,
    ) -> Bound<'py, PyArray1<i32>> {

        let local =  x.as_array().to_owned().to_vec();

        let comp = exp_golomb::decode_k_expgolomb_list(&local, count, k);

        PyArray1::from_vec(py, comp).to_owned()
    }


    // Entropy Coding
    #[pyfn(m)]
    #[pyo3(name = "compress_residuals")]
    fn compress_residuals<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, i32>,
        _tail_cut: f64,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {

        let local =  x.as_array();

        let comp = entropy::compress_residuals_rice(local).unwrap();

        Ok(PyArray1::from_vec(py, comp).to_owned())
    }

    #[pyfn(m)]
    #[pyo3(name = "decompress_residuals")]
    fn decompress_residuals<'py>(
        py: Python<'py>,
        compressed: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {

        let local = compressed.to_owned();
        let buf = local.as_slice().unwrap();
        let recon = entropy::decompress_residuals_rice(buf).unwrap();

        Ok(recon.to_pyarray(py))
    }

    Ok(())
}