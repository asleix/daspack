# tests/test_daspack.py
import numpy as np
import pytest

from daspack import DASCoder, Quantizer


def quantize_uniform_ref(x: np.ndarray, step: float) -> np.ndarray:
    """Reference quantizer used to check lossy roundtrip."""
    return np.round(x / step) * step


@pytest.mark.parametrize(
    "shape,block,levels,order,step",
    [
        ((3, 5), (2, 3), 1, 1, 0.5),
        ((32, 16), (8, 8), 2, 2, 0.25),
        ((64, 64), (16, 16), 1, 3, 0.1),
    ],
)
def test_roundtrip_uniform(shape, block, levels, order, step):
    rng = np.random.default_rng(0xC0FFEE)
    data = rng.uniform(-100, 100, size=shape).astype(np.float64)

    coder = DASCoder(threads=2)
    stream = coder.encode(
        data,
        Quantizer.Uniform(step=float(step)),
        blocksize=block,
        levels=levels,
        order=order,
    )
    assert isinstance(stream, (bytes, bytearray))
    assert len(stream) > 20

    out = coder.decode(stream)
    assert out.shape == shape
    assert out.dtype == np.float64

    expected = quantize_uniform_ref(data, step)
    np.testing.assert_allclose(out, expected, atol=step / 2 + 1e-12)


@pytest.mark.parametrize(
    "shape,block,levels,order",
    [
        ((4, 4), (2, 2), 0, 0),
        ((64, 128), (32, 32), 0, 0),
        ((128, 64), (16, 16), 0, 0),
    ],
)
def test_roundtrip_lossless(shape, block, levels, order):
    rng = np.random.default_rng(0xDEADBEEF)
    data = rng.integers(low=-1000, high=1000, size=shape, dtype=np.int32)

    coder = DASCoder(threads=4)
    stream = coder.encode(
        data,
        Quantizer.Lossless(),
        blocksize=block,
        levels=levels,
        order=order,
    )
    assert isinstance(stream, (bytes, bytearray))
    assert len(stream) > 4

    out = coder.decode(stream)
    assert out.shape == shape
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, data)


def test_decode_dtype_inference():
    # Uniform -> float64
    data_f = np.array([[0.2, -1.3], [3.7, 5.5]], dtype=np.float64)
    coder = DASCoder(threads=1)
    s_f = coder.encode(data_f, Quantizer.Uniform(step=0.5))
    out_f = coder.decode(s_f)
    assert out_f.dtype == np.float64

    # Lossless -> int32
    data_i = np.array([[1, -2], [3, -4]], dtype=np.int32)
    s_i = coder.encode(data_i, Quantizer.Lossless())
    out_i = coder.decode(s_i)
    assert out_i.dtype == np.int32


def test_stream_changes_when_step_changes():
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    coder = DASCoder(threads=1)
    s1 = coder.encode(data, Quantizer.Uniform(step=0.1))
    s2 = coder.encode(data, Quantizer.Uniform(step=0.2))
    assert s1 != s2


def test_encode_uniform_requires_float64():
    coder = DASCoder(threads=1)
    bad = np.array([[1, 2], [3, 4]], dtype=np.int32)
    with pytest.raises(Exception):
        coder.encode(bad, Quantizer.Uniform(step=0.5))


def test_encode_lossless_requires_int32():
    coder = DASCoder(threads=1)
    bad = np.array([[1.0, 2.0]], dtype=np.float64)
    with pytest.raises(Exception):
        coder.encode(bad, Quantizer.Lossless())


def test_uniform_roundtrip_tolerance_edge():
    # Values exactly on .5 step boundaries should survive de/quantization cleanly.
    step = 0.5
    vals = np.array(
        [[-1.0, -0.5, 0.0, 0.5, 1.0], [1.5, 2.0, 2.5, 3.0, 3.5]],
        dtype=np.float64,
    )
    coder = DASCoder(threads=1)
    s = coder.encode(vals, Quantizer.Uniform(step=step))
    out = coder.decode(s)
    expected = quantize_uniform_ref(vals, step)
    np.testing.assert_allclose(out, expected, atol=step / 2 + 1e-12)


@pytest.mark.parametrize("threads", [1, 2, 8])
def test_threads_param_does_not_change_result_uniform(threads):
    rng = np.random.default_rng(123)
    data = rng.normal(size=(32, 32)).astype(np.float64)
    coder = DASCoder(threads=threads)
    s = coder.encode(
        data, Quantizer.Uniform(step=0.25), blocksize=(8, 8), levels=1, order=2
    )
    out = coder.decode(s)
    expected = quantize_uniform_ref(data, 0.25)
    np.testing.assert_allclose(out, expected, atol=0.125 + 1e-12)


@pytest.mark.parametrize("threads", [1, 2, 8])
def test_threads_param_does_not_change_result_lossless(threads):
    rng = np.random.default_rng(456)
    data = rng.integers(-50, 50, size=(16, 16), dtype=np.int32)
    coder = DASCoder(threads=threads)
    s = coder.encode(
        data, Quantizer.Lossless(), blocksize=(8, 8), levels=0, order=0
    )
    out = coder.decode(s)
    np.testing.assert_array_equal(out, data)
