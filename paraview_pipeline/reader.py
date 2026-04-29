"""
CST HDF5 field-result reader.

Supported formats
-----------------
structured       E/H-Field shape (nz, ny, nx)       — rectilinear FIT/FEM, freq domain
structured_time  E/H-Field shape (nT, nz, ny, nx)   — structured time-domain monitor
unstructured     E/H-Field shape (N,)                — open-boundary or FEM point cloud
time_animation   E/H-Field shape (nT, N)             — unstructured time-domain monitor
"""
import h5py
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _field_key(h5):
    for k in h5.keys():
        if "Field" in k:
            return k
    raise KeyError("No E/H-Field dataset found in HDF5 file")


def detect_format(filepath):
    """Return 'structured', 'structured_time', 'unstructured', or 'time_animation'."""
    with h5py.File(filepath, "r") as f:
        shape = f[_field_key(f)].shape
    if len(shape) == 4:
        return "structured_time"
    if len(shape) == 3:
        return "structured"
    if len(shape) == 2:
        return "time_animation"
    if len(shape) == 1:
        return "unstructured"
    raise ValueError(f"Unrecognised field shape: {shape}")


def get_structured_shape(filepath):
    """Return (nz, ny, nx) without reading field data.

    Works for both 'structured' (nz, ny, nx) and 'structured_time'
    (nT, nz, ny, nx) — always returns the spatial dimensions only.
    """
    with h5py.File(filepath, "r") as f:
        return f[_field_key(f)].shape[-3:]


def get_time_steps(filepath):
    """Return (nT,) Times array for time-animation files, or None."""
    with h5py.File(filepath, "r") as f:
        if "Times" in f:
            return f["Times"][:]
    return None


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StructuredResult:
    x: np.ndarray          # (nx,) float32 mesh coordinates
    y: np.ndarray          # (ny,) float32
    z: np.ndarray          # (nz,) float32
    Ex: np.ndarray         # (nz, ny, nx) complex or real float32
    Ey: np.ndarray
    Ez: np.ndarray
    field_type: str        # 'E' or 'H'
    is_complex: bool


@dataclass
class UnstructuredResult:
    positions: np.ndarray  # (N, 3) float32
    Fx: np.ndarray         # (N,) complex or real float32
    Fy: np.ndarray
    Fz: np.ndarray
    field_type: str
    is_complex: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_components(arr):
    """Decompose compound-dtype array into (Fx, Fy, Fz, is_complex)."""
    dt = arr.dtype
    if dt["x"].names and "re" in dt["x"].names:
        return (
            arr["x"]["re"].astype(np.float32) + 1j * arr["x"]["im"].astype(np.float32),
            arr["y"]["re"].astype(np.float32) + 1j * arr["y"]["im"].astype(np.float32),
            arr["z"]["re"].astype(np.float32) + 1j * arr["z"]["im"].astype(np.float32),
            True,
        )
    return (
        arr["x"].astype(np.float32),
        arr["y"].astype(np.float32),
        arr["z"].astype(np.float32),
        False,
    )


def _build_selection(shape_full, subsample, slice_axis, slice_idx,
                     x_all, y_all, z_all):
    """
    Return (sel, x, y, z, iz_list, iy_sel, ix_sel) for z-slab streaming.
    sel is a tuple of slices suitable for direct h5 indexing.
    iz_list / iy_sel / ix_sel are used by the streaming writer.
    """
    nz_f, ny_f, nx_f = shape_full
    s = subsample

    if slice_axis == "z":
        i = slice_idx
        sel = (slice(i, i + 1), slice(None, None, s), slice(None, None, s))
        x, y, z = x_all[::s], y_all[::s], x_all[i:i+1]  # placeholder; z fixed below
        z = z_all[i:i + 1]
        iz_list = [i]
        iy_sel = slice(None, None, s)
        ix_sel = slice(None, None, s)
    elif slice_axis == "y":
        i = slice_idx
        sel = (slice(None, None, s), slice(i, i + 1), slice(None, None, s))
        x, y, z = x_all[::s], y_all[i:i + 1], z_all[::s]
        iz_list = list(range(0, nz_f, s))
        iy_sel = slice(i, i + 1)
        ix_sel = slice(None, None, s)
    elif slice_axis == "x":
        i = slice_idx
        sel = (slice(None, None, s), slice(None, None, s), slice(i, i + 1))
        x, y, z = x_all[i:i + 1], y_all[::s], z_all[::s]
        iz_list = list(range(0, nz_f, s))
        iy_sel = slice(None, None, s)
        ix_sel = slice(i, i + 1)
    else:
        sel = (slice(None, None, s), slice(None, None, s), slice(None, None, s))
        x, y, z = x_all[::s], y_all[::s], z_all[::s]
        iz_list = list(range(0, nz_f, s))
        iy_sel = slice(None, None, s)
        ix_sel = slice(None, None, s)

    return sel, x, y, z, iz_list, iy_sel, ix_sel


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def read_structured(filepath, subsample=1, slice_axis=None, slice_idx=0):
    """
    Load a structured rectilinear result into a StructuredResult.

    Parameters
    ----------
    subsample  : keep every Nth point in all axes (1 = no subsampling)
    slice_axis : 'x', 'y', 'z', or None for full 3-D
    slice_idx  : grid index along slice_axis

    For large files (>4 GB) use write_vtr_streaming() instead of this reader;
    it bypasses RAM by streaming z-slabs directly to disk.
    """
    with h5py.File(filepath, "r") as f:
        fkey = _field_key(f)
        ft = "E" if "E-Field" in fkey else "H"
        x_all = f["Mesh line x"][:].astype(np.float32)
        y_all = f["Mesh line y"][:].astype(np.float32)
        z_all = f["Mesh line z"][:].astype(np.float32)

        sel, x, y, z, _, _, _ = _build_selection(
            f[fkey].shape, subsample, slice_axis, slice_idx, x_all, y_all, z_all
        )
        E = f[fkey][sel]          # HDF5 reads only the selected region

    Ex, Ey, Ez, is_complex = _extract_components(E)
    return StructuredResult(
        x=x, y=y, z=z, Ex=Ex, Ey=Ey, Ez=Ez,
        field_type=ft, is_complex=is_complex,
    )


def read_structured_time(filepath, time_idx=None, subsample=1,
                         slice_axis=None, slice_idx=0):
    """
    Load one time step from a structured_time (nT, nz, ny, nx) result.

    Parameters
    ----------
    time_idx   : which time step to load (default 0)
    subsample, slice_axis, slice_idx — same semantics as read_structured

    Returns a StructuredResult (is_complex=False for time-domain data).
    Use get_time_steps() to retrieve the Times array.
    """
    ti = 0 if time_idx is None else time_idx
    with h5py.File(filepath, "r") as f:
        fkey = _field_key(f)
        ft = "E" if "E-Field" in fkey else "H"
        x_all = f["Mesh line x"][:].astype(np.float32)
        y_all = f["Mesh line y"][:].astype(np.float32)
        z_all = f["Mesh line z"][:].astype(np.float32)

        shape_full = f[fkey].shape[1:]   # (nz, ny, nx) — drop time axis
        sel, x, y, z, _, _, _ = _build_selection(
            shape_full, subsample, slice_axis, slice_idx, x_all, y_all, z_all
        )
        E = f[fkey][(ti,) + sel]         # prepend time index to spatial sel

    Ex, Ey, Ez, is_complex = _extract_components(E)
    return StructuredResult(
        x=x, y=y, z=z, Ex=Ex, Ey=Ey, Ez=Ez,
        field_type=ft, is_complex=is_complex,
    )


def read_unstructured(filepath, time_idx=None):
    """
    Load an unstructured or time-animation result.

    For time_animation files, time_idx selects which frame to load (default 0).
    """
    with h5py.File(filepath, "r") as f:
        fkey = _field_key(f)
        ft = "E" if "E-Field" in fkey else "H"
        shape = f[fkey].shape
        pos_raw = f["Position"][:]

        if len(shape) == 1:
            F = f[fkey][:]
        else:                          # (nT, N) time animation
            ti = 0 if time_idx is None else time_idx
            F = f[fkey][ti, :]

    positions = np.column_stack(
        [pos_raw["x"], pos_raw["y"], pos_raw["z"]]
    ).astype(np.float32)
    Fx, Fy, Fz, is_complex = _extract_components(F)
    return UnstructuredResult(
        positions=positions, Fx=Fx, Fy=Fy, Fz=Fz,
        field_type=ft, is_complex=is_complex,
    )
