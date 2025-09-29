import os
import time
from dataclasses import asdict
from typing import Iterable, Mapping, Any
import numpy as np
from numpy.typing import NDArray
from tempfile import NamedTemporaryFile

# helper functions
from compression_methods import CodecOptions, Codec


def _file_size_bytes(path: str) -> int:
    return os.path.getsize(path)


def _verify_arrays(
    original: NDArray,
    restored: NDArray,
    opts: CodecOptions,
) -> tuple[bool, dict[str, float]]:
    """
    For lossless: exact match (shape, dtype, values).
    For lossy: allow absolute error up to quant_step using allclose.
    Returns (ok, metrics).
    """
    metrics: dict[str, float] = {}
    if opts.lossless:
        ok = (
            original.shape == restored.shape
            and original.dtype == restored.dtype
            and np.array_equal(original, restored)
        )
        # still compute error metrics for visibility (will be 0 if ok)
        diff = restored.astype(np.float64) - original.astype(np.float64)
        metrics["max_abs_err"] = (
            float(np.max(np.abs(diff))) if diff.size else 0.0
        )
        metrics["mse"] = float(np.mean(diff**2)) if diff.size else 0.0
        return ok, metrics
    else:
        # Lossey: accept atol = quant_step (absolute error tolerance).
        atol = float(opts.quant_step)
        # Shapes must match for meaningful comparison.
        if original.shape != restored.shape:
            return False, {"max_abs_err": np.inf, "mse": np.inf}
        ok = np.allclose(
            restored.astype(np.float64),
            original.astype(np.float64),
            rtol=0.0,
            atol=atol,
            equal_nan=True,
        )
        diff = restored.astype(np.float64) - original.astype(np.float64)
        metrics["max_abs_err"] = (
            float(np.max(np.abs(diff))) if diff.size else 0.0
        )
        metrics["mse"] = float(np.mean(diff**2)) if diff.size else 0.0
        return ok, metrics


def benchmark_once(
    codec: Codec,
    arr: NDArray,
    opts: CodecOptions,
    verify: bool = True,
) -> dict[str, Any]:
    """
    Run a single benchmark:
      - writes arr using codec+opts to a temp HDF5 file
      - measures write/read times and compressed file size
      - computes compression ratio (original_nbytes / file_size)
      - optional verification (exact for lossless, allclose with atol=quant_step for lossy)
    Returns a dict row with metrics.
    """
    # pick a temp file name (we want the file to persist during read)
    with NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_name = tmp.name

    original_bytes = int(arr.nbytes)
    write_s = time.perf_counter()
    codec.write(arr, tmp_name, opts)
    write_t = time.perf_counter() - write_s

    read_s = time.perf_counter()
    restored = codec.read(tmp_name)
    read_t = time.perf_counter() - read_s

    compressed_bytes = _file_size_bytes(tmp_name)
    # Clean up the temp file
    try:
        os.remove(tmp_name)
    except OSError:
        pass

    # Compression factor: how many times smaller the file is vs original in-memory array
    # (factor >= 1 means compression; < 1 means expansion)
    compression_factor = (
        (original_bytes / compressed_bytes)
        if compressed_bytes > 0
        else float("inf")
    )

    row: dict[str, Any] = {
        "codec": getattr(codec, "name", codec.__class__.__name__),
        "lossless": bool(opts.lossless),
        "quant_step": float(opts.quant_step),
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_factor": compression_factor,
        "write_seconds": write_t,
        "read_seconds": read_t,
    }

    if verify:
        ok, err = _verify_arrays(arr, restored, opts)
        row["verified"] = bool(ok)
        row.update(err)  # adds max_abs_err, mse

    return row


def run_benchmarks(
    codecs: Iterable[Codec],
    datasets: Mapping[str, NDArray],
    options: Iterable[CodecOptions],
    verify: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Orchestrates benchmarks for each (codec, dataset, options) triple.
    Returns a list of row dicts (one per run).
    If pandas is installed, you can turn it into a DataFrame via `to_dataframe(rows)`.
    """
    rows: list[dict[str, Any]] = []
    for ds_name, arr in datasets.items():
        for codec in codecs:
            for opts in options:
                if verbose:
                    print(
                        f"Running benchmark: dataset='{ds_name}', codec='{codec.name}', options={opts}"
                    )
                row = benchmark_once(codec, arr, opts, verify=verify)
                row["dataset"] = ds_name
                rows.append(row)
    return rows
