//! src/lib.rs – “HDF5 DASPack 0.1” filter plugin
//! -------------------------------------------------------------
//! Build with `cargo build --release --lib`
//! Add the resulting shared library to HDF5_PLUGIN_PATH
//! -------------------------------------------------------------

mod wavelets;
mod prediction;
pub mod entropy;
mod blocks;
pub use crate::entropy::{compress_residuals_rice, decompress_residuals_rice};
pub use crate::blocks::{compress_lossless, decompress_lossless, CompressParams};

pub use crate::entropy::exp_golomb::encode_k_expgolomb_list;


#[cfg(feature = "python")]
mod pybindings;

#[cfg(feature = "hdf5")]
mod hdf5_plugin;
#[cfg(feature = "hdf5")]
pub use hdf5_plugin::*;


