# daspack/__init__.py

import os
import ctypes
import h5py


def _register_daspack_filter():
    # 1) Locate the shared‐object right next to this file
    pkg_dir = os.path.dirname(__file__)
    so_name = next(
        fn
        for fn in os.listdir(pkg_dir)
        if fn.startswith("compute.") and fn.endswith(".so")
    )
    so_path = os.path.join(pkg_dir, so_name)

    # 2) Load it and grab the plugin‐info struct pointer
    lib = ctypes.CDLL(so_path)
    lib.H5PLget_plugin_info.restype = ctypes.c_void_p
    struct_ptr = lib.H5PLget_plugin_info()

    # 3) Register with HDF5’s C‐API
    #    (h5py.h5z.register_filter is a thin wrapper around H5Zregister)
    h5py.h5z.register_filter(struct_ptr)


# Call it immediately on import
try:
    _register_daspack_filter()
except Exception as e:
    # If something goes wrong, warn rather than hard-fail
    import warnings

    warnings.warn(f"Could not register DASPack filter: {e!r}", stacklevel=2)


def get_filter():
    return 33000
