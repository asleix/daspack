//! src/lib.rs – “HDF5 DASPack 0.1” filter plugin
//! -------------------------------------------------------------
//! Build with `cargo build --release --lib`
//! Add the resulting shared library to HDF5_PLUGIN_PATH
//! -------------------------------------------------------------

pub mod core;
pub mod codec;

pub use crate::core::entropy::{compress_residuals_rice, decompress_residuals_rice};
pub use crate::codec::{DASCoder, CompressParams, UniformQuantizer, LosslessQuantizer};


#[cfg(feature = "python")]
mod pybindings;

#[cfg(feature = "hdf5")]
mod hdf5_plugin;

#[cfg(feature = "hdf5")]
pub use hdf5_plugin::*;


