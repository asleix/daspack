
// hdf5_plugin.rs

use crate::codec::{CodecLossless, Codec, CompressParams};


// ───────────── dependencies ─────────────
use std::{
    ffi::c_void,
    os::raw::{c_int, c_uint, c_char},
    ptr,
    slice,
};

use anyhow::{Context, Result};
use hdf5_sys::{
    h5i::hid_t,
    h5p::{H5Pget_chunk},
    h5pl::H5PL_TYPE_FILTER,
    h5s::{ H5Sget_simple_extent_ndims},
    h5t::{H5Tget_class, H5Tget_size, H5T_INTEGER},
    h5z::{
        H5Z_class2_t, H5Z_filter_t, H5Z_FLAG_REVERSE, H5Z_CLASS_T_VERS,
    },
};
use numpy::ndarray::{ Array2};

//────────────── filter identification ─────────────
const DASPACK_FILTER_ID: H5Z_filter_t = 33_000;
const DASPACK_NAME: &[u8] = b"DASPack 0.1\0";

//────────────── helper: read params with defaults ─────────────
#[derive(Clone, Debug)]
struct RuntimeParams {
    rows: usize,
    cols: usize,
    cparams: CompressParams,
    is_be: bool
}

impl RuntimeParams {
    /// cd_values layout (all `u32`)
    /// [0] version
    /// [1] rows in chunk
    /// [2] cols in chunk
    /// [3] element size  (always 4)
    /// [4] block_height
    /// [5] block_width
    /// [6] lx
    /// [7] lt
    /// [8] lpc_order
    /// [10] byte-order  (0 = little, 1 = big)
    unsafe fn from_cd_values(cd_vals: &[c_uint]) -> Self {
        let mut it = cd_vals.iter().copied();
        let _ver  = it.next().unwrap_or(1);
        let rows  = it.next().unwrap_or(0) as usize;
        let cols  = it.next().unwrap_or(0) as usize;
        let _esz  = it.next().unwrap_or(4);

        let bh    = it.next().unwrap_or(64) as usize;
        let bw    = it.next().unwrap_or(64) as usize;
        let lx    = it.next().unwrap_or(1)  as usize;
        let lt    = it.next().unwrap_or(1)  as usize;
        let lpc   = it.next().unwrap_or(4)  as usize;
        let is_be    = it.next().unwrap_or(0) != 0;

        let mut cparams = CompressParams::new(bh, bw, lx, lt, lpc);
        cparams.lpc_bits = 8;                 // keep defaults
        cparams.lpc_range = (-1.0, 1.0);

        Self { rows, cols, cparams, is_be}
    }
}


//────────────── 1. plugin registration ─────────────

#[allow(non_upper_case_globals)]
static mut DASPACK_CLASS: H5Z_class2_t = H5Z_class2_t {
    version:          H5Z_CLASS_T_VERS as c_int,
    id:               DASPACK_FILTER_ID,
    encoder_present:  1,
    decoder_present:  1,
    name:             DASPACK_NAME.as_ptr() as *const c_char,
    can_apply:  Some(daspack_can_apply),
    set_local: Some(daspack_set_local),
    filter:    Some(daspack_filter),
};

#[unsafe(no_mangle)]
pub extern "C" fn H5PLget_plugin_type() -> c_int {
    H5PL_TYPE_FILTER as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn H5PLget_plugin_info() -> *const c_void {
    // `addr_of!` gets a raw pointer without creating `&` or `&mut`
     ptr::addr_of!(DASPACK_CLASS) as *const c_void 
}

//────────────── 2. can_apply – 2-D int32 only ─────────────
extern "C" fn daspack_can_apply(_dcpl: hid_t, type_id: hid_t, space_id: hid_t) -> c_int {
    unsafe {
        // dataset must be 2-D
        let ndims = H5Sget_simple_extent_ndims(space_id);
        if ndims != 2 {
            eprintln!("DASPack: only 2-D datasets supported (got {ndims}-D)");
            return 0;
        }
        // datatype must be 32-bit integer
        if H5Tget_class(type_id) != H5T_INTEGER {
            eprintln!("DASPack: supports only integer types");
            return 0;
        }
        if H5Tget_size(type_id) != 4 {
            eprintln!("DASPack: supports only 32-bit ints (got {} bytes)", H5Tget_size(type_id));
            return 0;
        }
        1 // ok
    }
}

extern "C" fn daspack_set_local(dcpl: hid_t, type_id: hid_t, space: hid_t) -> c_int {
    unsafe {
        use hdf5_sys::{
            h5p::{H5Pget_filter_by_id2, H5Pmodify_filter},
            h5t::{H5Tget_order, H5T_ORDER_BE, H5T_ORDER_LE},
        };

        // ---- dataset & chunk geometry ----------------------------------
        let ndims = H5Sget_simple_extent_ndims(space) as usize;
        let mut chunk = vec![0u64; ndims];
        H5Pget_chunk(dcpl, ndims as c_int, chunk.as_mut_ptr());
        let (rows, cols) = (chunk[0] as usize, chunk[1] as usize);

        // ---- fetch any user-supplied compression_opts ------------------
        let mut flags: c_uint = 0;
        let mut cd_len: size_t = 20;
        let mut cd_buf = [0u32; 20];

        H5Pget_filter_by_id2(
            dcpl,
            DASPACK_FILTER_ID,
            &mut flags,
            &mut cd_len,
            cd_buf.as_mut_ptr(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        let user_opts = &cd_buf[..cd_len];

        // merge 
        let default = CompressParams::new(64, 64, 1, 1, 4);
        // final cd_values array
        let merged: [u32; 5] = match user_opts.len() {
            5 => {
                // Try converting; if it fails, report and error out
                match user_opts.try_into() {
                    Ok(arr) => arr,
                    Err(_) => {
                        eprintln!("DASPack: unexpected compression_opts length {} (expected 5)", user_opts.len());
                        return -1;
                    }
                }
            }
            0 => [
                default.block_height  as u32,
                default.block_width   as u32,
                default.lx            as u32,
                default.lt            as u32,
                default.lpc_order     as u32,
            ],
            len => {
                eprintln!("DASPack: compression_opts must have exactly 5 ints (got {})", len);
                return -1;
            }
        };


        // byte order flag 
        let order_flag = match H5Tget_order(type_id) {
            H5T_ORDER_LE => 0,
            H5T_ORDER_BE => 1,
            _ => {
                eprintln!("DASPack: unsupported byte order");
                return -1;
            }
        };

        // final cd_values array
        let cd_vec: Vec<u32> = [
            &[1, rows as u32, cols as u32, 4][..],
            &merged,
            &[order_flag],
        ]
        .concat();

        // write back
        let status = H5Pmodify_filter(
            dcpl,
            DASPACK_FILTER_ID,
            flags,
            cd_vec.len(),
            cd_vec.as_ptr(),
        );
        if status < 0 {
            eprintln!("DASPack: H5Pmodify_filter failed");
            return -1;
        }
        0
    }
}

//────────────── filter – compression & decompression ────
extern "C" fn daspack_filter(
    flags: c_uint,
    cd_nelmts: size_t,
    cd_values: *const c_uint,
    nbytes: size_t,
    buf_size: *mut size_t,
    buf: *mut *mut c_void,
) -> size_t {
    unsafe {
        //-------------------- client-data slice ---------------------------
        let cd_vals: &[c_uint] = if cd_nelmts == 0 || cd_values.is_null() {
            eprintln!("cd values are empty");
            &[]               // use defaults later
        } else {
            // SAFETY: cd_values comes from HDF5; assume alignment ok.
            // The len check prevents overflow panic.
            slice::from_raw_parts(cd_values, cd_nelmts)
        };
        let params = RuntimeParams::from_cd_values(cd_vals);

        //-------------------- input chunk slice --------------------------
        let in_ptr = *buf as *const u8;
        let input: &[u8] = if nbytes == 0 || in_ptr.is_null() {
            eprintln!("input chunk slice is empty");
            &[]
        } else {
            // SAFETY: HDF5 guarantees `nbytes` is the size of the memory block.
            // We also cap it to isize::MAX just in case (should never happen).
            if nbytes > isize::MAX as usize {
                eprintln!("DASPack: chunk larger than isize::MAX - refusing");
                return 0;
            }
            slice::from_raw_parts(in_ptr, nbytes)
        };

        //-----------------------------------------------------------------

        let is_decode = (flags & H5Z_FLAG_REVERSE) != 0;

        // Run (de)compression and capture any error
        let result: Result<Vec<u8>> = if is_decode {
            decode_chunk(input, &params)
        } else {
            encode_chunk(input, &params)
        };

        let out = match result {
            Ok(v) => v,
            Err(e) => {
                eprintln!("DASPack filter failure: {e:?}");
                return 0; // signal error to HDF5
            }
        };

        // allocate HDF5-owned buffer for result
        let out_len = out.len();
        let out_ptr = libc::malloc(out_len);
        if out_ptr.is_null() {
            eprintln!("DASPack: malloc failed");
            return 0;
        }
        ptr::copy_nonoverlapping(out.as_ptr(), out_ptr as *mut u8, out_len);

        // hand result back to HDF5
        *buf = out_ptr;
        *buf_size = out_len;
        out_len
    }
}

//────────────── helpers – transpose raw - ndarray ─────────
#[allow(non_camel_case_types)]
type size_t = usize;

fn encode_chunk(raw: &[u8], p: &RuntimeParams) -> Result<Vec<u8>> {
    let need = p.rows * p.cols * 4;
    anyhow::ensure!(raw.len() == need,
        "raw chunk size {} != rows*cols*4 {}", raw.len(), need);

    let mut vals = Vec::<i32>::with_capacity(p.rows * p.cols);
    for bytes in raw.chunks_exact(4) {
        let v = if p.is_be {
            i32::from_be_bytes(bytes.try_into().unwrap())
        } else {
            i32::from_le_bytes(bytes.try_into().unwrap())
        };
        vals.push(v);
    }

    let arr = Array2::from_shape_vec((p.rows, p.cols), vals)
        .context("shape mismatch while forming ndarray")?;

    let codec = CodecLossless::new(p.cparams.clone(), 0)?;
    codec.compress(arr.view())
}


fn decode_chunk(comp: &[u8], p: &RuntimeParams) -> Result<Vec<u8>> {
    let codec = CodecLossless::new(p.cparams.clone(), 0)?;
        
    let arr = codec.decompress(comp, (p.rows, p.cols))?;
    let mut out = Vec::<u8>::with_capacity(p.rows * p.cols * 4);

    for v in arr.iter() {
        if p.is_be {
            out.extend_from_slice(&v.to_be_bytes());
        } else {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}