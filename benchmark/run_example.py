import numpy as np
import h5py

from compression_methods import CodecOptions, Zfp, Gzip, JPEG2000, DASPack
from bench_helper import run_benchmarks

# Suppose you’ve defined your codecs:
codecs = [Zfp(), Gzip(), JPEG2000(), DASPack()]


def load_hdas_segment(h5_path: str):
    """Return (data_raw, dt_s, begin_time, dx_m) from HDAS .h5 file."""
    with h5py.File(h5_path, "r") as f:
        # convert to float32 for accurate compression factors
        data_raw = np.array(f["data"][:]).astype(np.float32)
        dt_s = float(f["data"].attrs["dt_s"])
        begin_iso = f["data"].attrs["begin_time"]
        dx_m = float(f["data"].attrs["dx_m"])
    return data_raw, dt_s, begin_iso, dx_m


# Datasets you want to try:
datasets = {
    "alboran": load_hdas_segment("data/7M_S_20231005-214004_RAW.h5")[0],
    "castor": load_hdas_segment("data/ZI_C_20230609-102630_RAW.h5")[0],
}

# Configurations:
opts_list = [
    CodecOptions.create_with_quant_step(quant_step=0.01),
    CodecOptions.create_with_quant_step(quant_step=0.1),
]

# Run
rows = run_benchmarks(codecs, datasets, opts_list, verify=True, verbose=True)

# Optional: view as a DataFrame if pandas is available
import pandas as pd

df = pd.DataFrame(rows)
print(df.sort_values(["dataset", "codec", "quant_step"]))
df.to_csv("benchmark_results.csv")
