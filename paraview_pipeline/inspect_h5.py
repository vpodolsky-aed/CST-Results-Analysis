"""Standalone HDF5 inspector — prints shape, dtype, and size of every .h5 in BASE."""
import h5py
import os

BASE = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\Field dump in HDF5"

files = sorted(f for f in os.listdir(BASE) if f.endswith(".h5"))
for fn in files:
    path = os.path.join(BASE, fn)
    sz = os.path.getsize(path) / 1e6
    with h5py.File(path, "r") as f:
        shapes = {k: f[k].shape for k in f.keys() if hasattr(f[k], "shape")}
        field_key = next((k for k in f.keys() if "Field" in k), None)
        dtype_info = str(f[field_key].dtype) if field_key else "n/a"
    print(f"{fn}  ({sz:.1f} MB)")
    for k, sh in shapes.items():
        print(f"  {k}: {sh}")
    print(f"  field dtype: {dtype_info}")
    print()
