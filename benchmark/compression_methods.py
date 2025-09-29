from abc import ABC
from typing import ClassVar
from numpy.typing import NDArray
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
import h5py
import numpy as np


@dataclass
class CodecOptions:
    lossless: bool
    quant_step: float

    @classmethod
    def create_lossless(cls):
        return cls(True, 0.0)

    @classmethod
    def create_with_quant_step(cls, quant_step: float):
        if quant_step <= 0:
            raise ValueError("quant_step must be > 0 for lossy modes")
        return cls(False, float(quant_step))


class Codec(ABC):
    name: ClassVar[str]

    def write(self, arr: NDArray, fn: str, opts: CodecOptions) -> None: ...
    def read(self, fn: str) -> NDArray: ...


class DASPack(Codec):
    name: ClassVar[str] = "daspack"

    def write(self, arr: NDArray, fn: str, opts: CodecOptions) -> None:
        from daspack import DASCoder, Quantizer

        if opts.lossless:
            quantizer = Quantizer.Lossless()
        else:
            quantizer = Quantizer.Uniform(opts.quant_step)
            arr = arr.astype(np.float64)

        coder = DASCoder(threads=1)
        buf = coder.encode(arr, quantizer)

        with h5py.File(fn, "w") as hf:
            dset = hf.create_dataset(
                "compressed", data=np.frombuffer(buf, dtype=np.uint8)
            )
            dset.attrs["lossless"] = opts.lossless
            dset.attrs["quant_step"] = float(opts.quant_step)

    def read(self, fn: str) -> NDArray:
        from daspack import DASCoder

        with h5py.File(fn) as hf:
            raw = hf["compressed"][:].tobytes()

        coder = DASCoder(threads=4)
        restored = coder.decode(raw)  # dtype & shape are in the stream
        return restored


@dataclass
class Zfp(Codec):
    name: ClassVar[str] = "zfp"

    def write(self, arr: NDArray, fn: str, opts) -> None:
        import hdf5plugin

        with h5py.File(fn, "w") as f:
            if opts.lossless:
                dset = f.create_dataset(
                    "data",
                    data=arr,
                    compression=hdf5plugin.Zfp(reversible=True),
                )
            else:
                dset = f.create_dataset(
                    "data",
                    data=arr,
                    compression=hdf5plugin.Zfp(
                        accuracy=float(opts.quant_step)
                    ),
                )
            dset.attrs["lossless"] = bool(opts.lossless)
            dset.attrs["quant_step"] = float(opts.quant_step)

    def read(self, fn: str) -> NDArray:
        with h5py.File(fn) as f:
            # ZFP plugin transparently decompresses back to the original array dtype/shape.
            return f["data"][:]


def convert_to_integer_tolerance(data: NDArray, tol=1.0):
    if tol <= 0:
        raise ValueError("tol must be > 0")
    rounded = np.round(data / tol)
    return rounded.astype(np.int64)


class Gzip(Codec):
    name: ClassVar[str] = "gzip"

    @staticmethod
    def _int_dtype_for_range(arr: NDArray):
        # Safe helper for empty arrays too
        if arr.size == 0:
            return np.int16
        max_abs = np.max(np.abs(arr))
        if max_abs == 0:
            return np.int16
        lg = np.log2(max_abs)
        if lg > 30:
            return np.int64
        elif lg > 15:
            return np.int32
        else:
            return np.int16

    def _write_int_arr(self, arr: NDArray, fn: str, opts: CodecOptions):
        data_type = self._int_dtype_for_range(arr)
        with h5py.File(fn, "w") as f:
            dset = f.create_dataset(
                "data",
                data=arr.astype(data_type, copy=False),
                compression="gzip",
                dtype=data_type,
            )
            dset.attrs["lossless"] = bool(opts.lossless)
            dset.attrs["quant_step"] = float(opts.quant_step)

    def write(self, arr: NDArray, fn: str, opts: CodecOptions) -> None:
        if opts.lossless:
            # Preserve dtype & values losslessly with gzip
            with h5py.File(fn, "w") as f:
                dset = f.create_dataset("data", data=arr, compression="gzip")
                dset.attrs["lossless"] = True
                dset.attrs["quant_step"] = 0.0
        else:
            # Quantize to integer grid, then gzip (lossless) those integers
            data_int = convert_to_integer_tolerance(
                arr, float(opts.quant_step)
            )
            self._write_int_arr(data_int, fn, opts)

    def read(self, fn: str) -> NDArray:
        with h5py.File(fn) as f:
            dset = f["data"]
            lossless = bool(dset.attrs.get("lossless", True))
            q = float(dset.attrs.get("quant_step", 0.0))
            data: NDArray = dset[:]

        if lossless or q == 0.0:
            # Return exactly what was written
            return data
        else:
            # De-quantize to floats
            return data.astype(np.float64) * q


class JPEG2000(Codec):
    name: ClassVar[str] = "jpeg2000"

    def _write_int_arr(self, arr: NDArray, fn: str, opts: CodecOptions):
        from imagecodecs import jpeg2k_encode

        # Choose an integer dtype appropriate for range
        if arr.size == 0:
            data_type = np.int16
        else:
            max_abs = np.max(np.abs(arr))
            if max_abs == 0:
                data_type = np.int16
            else:
                lg = np.log2(max_abs)
                if lg > 30:
                    data_type = np.int64
                elif lg > 15:
                    data_type = np.int32
                else:
                    data_type = np.int16

        encoded = jpeg2k_encode(
            arr.astype(data_type, copy=False),
            reversible=True,  # always reversible; "loss" (if any) is external quantization
            numthreads=1,
        )

        with h5py.File(fn, "w") as f:
            dset = f.create_dataset(
                "data", data=np.frombuffer(encoded, dtype=np.uint8)
            )
            # Persist metadata to reconstruct on read
            dset.attrs["lossless"] = bool(opts.lossless)
            dset.attrs["quant_step"] = float(opts.quant_step)

    def write(self, arr: NDArray, fn: str, opts: CodecOptions) -> None:
        if opts.lossless:
            self._write_int_arr(arr, fn, opts)
        else:
            data_int = convert_to_integer_tolerance(
                arr, float(opts.quant_step)
            )
            self._write_int_arr(data_int, fn, opts)

    def read(self, fn: str) -> NDArray:
        from imagecodecs import jpeg2k_decode

        with h5py.File(fn) as f:
            raw = f["data"][:].tobytes()
            attrs = f["data"].attrs
            lossless = bool(attrs.get("lossless", True))
            q = float(attrs.get("quant_step", 0.0))

        decoded = jpeg2k_decode(raw)

        if lossless or q == 0.0:
            return decoded
        else:
            # de-quantize back to float
            return decoded.astype(np.float64) * q
