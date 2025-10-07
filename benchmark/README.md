# Benchmarking Scripts

This folder contains scripts for running comparative benchmarks of **DASPack** and other compression methods.

## Data

Example datasets can be downloaded from the following DOI link:
[https://doi.org/10.20350/digitalCSIC/17586](https://doi.org/10.20350/digitalCSIC/17586)

Each dataset should be stored in the `data/` subdirectory (e.g., `data/7M_S_20231005-214004_RAW.h5`).

## Scripts Overview

- **`compression_methods.py`** — Defines the compression codecs and helper classes:
  - `CodecOptions`: Configuration options for lossless or quantized compression.
  - `DASPack`, `Zfp`, `Gzip`, and `JPEG2000`: Implementations of different codecs used in benchmarking.

- **`bench_helper.py`** — Contains helper functions used for running and analyzing benchmark tests. Includes utilities for verifying results and collecting metrics.

- **`run_example.py`** — The main entry point to run benchmarks. It provides an example workflow for:
  1. Loading datasets from `.h5` files.
  2. Defining codec configurations.
  3. Executing benchmarks.
  4. Exporting results to CSV.

## Example Usage

To run the benchmarks, modify `run_example.py` as needed (e.g., to point to your dataset files or adjust quantization steps), then execute:

```bash
python run_example.py
```

### Example Script Summary (`run_example.py`)

1. **Imports codecs and helpers:**
   ```python
   from compression_methods import CodecOptions, Zfp, Gzip, JPEG2000, DASPack
   from bench_helper import run_benchmarks
   ```

2. **Defines available codecs:**
   ```python
   codecs = [Zfp(), Gzip(), JPEG2000(), DASPack()]
   ```

3. **Loads HDAS datasets:**
   ```python
   def load_hdas_segment(h5_path: str):
       with h5py.File(h5_path, 'r') as f:
           data_raw = np.array(f['data'][:]).astype(np.float32)
           dt_s = float(f['data'].attrs['dt_s'])
           begin_iso = f['data'].attrs['begin_time']
           dx_m = float(f['data'].attrs['dx_m'])
       return data_raw, dt_s, begin_iso, dx_m
   ```

4. **Defines datasets and quantization settings:**
   ```python
   datasets = {
       'alboran': load_hdas_segment('data/7M_S_20231005-214004_RAW.h5')[0],
       'castor': load_hdas_segment('data/ZI_C_20230609-102630_RAW.h5')[0],
   }

   opts_list = [
       CodecOptions.create_with_quant_step(quant_step=0.01),
       CodecOptions.create_with_quant_step(quant_step=0.1),
   ]
   ```

5. **Runs the benchmarks and exports results:**
   ```python
   rows = run_benchmarks(codecs, datasets, opts_list, verify=True, verbose=True)

   import pandas as pd
   df = pd.DataFrame(rows)
   df.to_csv('benchmark_results.csv', index=False)
   ```

## Output

After execution, a file named `benchmark_results.csv` will be generated containing the benchmark results. The table includes information such as dataset name, codec type, quantization step, compression ratio, and reconstruction error.

## Notes

- All codecs rely on **HDF5** for file handling.
- `hdf5plugin` and `imagecodecs` must be installed to support ZFP and JPEG2000 codecs respectively.
- The DASPack codec requires the `daspack` Python package.

---

**Author:** Aleix Seguí  
**Purpose:** Provide a consistent benchmarking framework for evaluating DASPack against other compression standards.

