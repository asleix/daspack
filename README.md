<p align="center">
  <img src="docs/assets/logo.svg" alt="DASPack Logo" />
</p>

# DASPack: Controlled data compression for Distributed Acoustic Sensing

DASPack is a fast, open‑source compressor that lets you store huge Distributed Acoustic Sensing (DAS) datasets **losslessly** or with a user‑defined, fixed reconstruction error. It couples wavelets, linear predictive coding and arithmetic coding in Rust, and plugs straight into the HDF5 pipeline so you can write compressed datasets with a single `h5py` call.

---

## ✨ Highlights

- **Fixed‑accuracy**: pick an absolute error (including zero) and know exactly what you lose.
- **Real‑time throughput**: ≥ 800 MB s⁻¹ on an 8‑core laptop.
- **Seamless HDF5**: available as filter *33 000*; works with `h5py`, `h5netcdf`, MATLAB, …
- **Safe & portable**: core written in Rust, no unsafe C buffers in user code.
- **Python bindings**: one‑line `compress_data` / `decompress_data` helpers for quick prototyping.

---

## 🚀 Quick start

### 1. Install (Python ≥ 3.9)

```bash
pip install daspack               # pre‑built wheels for Linux/macOS x86_64 & Apple Silicon
# or, from source:
# cargo install --path .          # Rust toolchain ≥ 1.74 required
```

### 2. Write compressed data

```python
import numpy as np, h5py, daspack

data   = np.random.randint(-1000, 1000, size=(4096, 8192), dtype=np.int32)
fname  = "example.h5"
b_side = 1024   # block height & width
wav    = 1    # one wavelet level in time & space
lpc    = 2    # 2‑tap linear predictor

with h5py.File(fname, "w") as f:
    f.create_dataset(
        name="data",
        data=data,
        chunks=data.shape,         # one chunk per tile (tune as needed)
        dtype=np.int32,
        compression=daspack.get_filter(),
        compression_opts=(
            b_side,  # block_height
            b_side,  # block_width
            wav,     # lx (space levels)
            wav,     # lt (time levels)
            lpc,     # lpc_order
            250,     # tailcut ‰  (leave default)
        ),
    )
```

### 3. Read it back

```python
import h5py, daspack # needed to properly load the hdf5 decompressor

with h5py.File(fname) as f:
    x = f["data"][:]
```

The filter is self‑describing: readers that do not know DASPack will ignore it and still access the raw bytes (just without decompression).

---

## 🧩 API reference

### `daspack.get_filter()`

Returns the integer *33 000* required by HDF5 to tag the dataset. Call once and reuse.

### Python helpers

```python
import daspack
bitstream = daspack.compress_data(array_2d_int32)  # → 1‑D uint8 buffer
restored  = daspack.decompress_data(bitstream, array_2d_int32.shape)
```

Both helpers are thin wrappers around the Rust core; they allocate NumPy arrays and copy no data inside the Rust/Python boundary.

---

## ⚙️ How it works (one‑liner)

```
Rounding to ints → Wavelet (5/3) and 2‑D LPC → Arithmetic coding
```

Everything except the initial rounding is perfectly reversible. See the [paper](docs/about.md) for a deep dive.

---

## 📄 License

DASPack is released under the 3-Clause BSD License.

---

## 🤝 Contributing

Bug reports and pull requests are welcome! Please open an issue first if you plan a large change so we can discuss it.

---

## 📣 Citing

If you use DASPack in academic work, please cite:

> Seguí, A. *et al.* (2025). **DASPack: Controlled Data Compression for Distributed Acoustic Sensing**. *Geophysical Journal International*.\
> DOI: *pending*

Thanks for supporting open science!

