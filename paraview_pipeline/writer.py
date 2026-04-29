"""
VTK file writers for ParaView.

write_vtr()           — RectilinearGrid from StructuredResult (in-memory)
write_vtr_streaming() — RectilinearGrid directly from HDF5, z-slab by z-slab
                        (safe for files too large to load into RAM;
                         accepts time_idx for structured_time 4-D files)
write_vtu()           — UnstructuredGrid from UnstructuredResult (point cloud)
write_pvd()           — ParaView Collection file (.pvd) for animation
"""
import struct
import tempfile
import numpy as np


# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------

def _hdr(nbytes: int) -> bytes:
    """UInt64 byte-count header for VTK appended-raw format."""
    return struct.pack("<Q", int(nbytes))


def _block(arr: np.ndarray, dtype=np.float32) -> bytes:
    raw = arr.astype(dtype).tobytes()
    return _hdr(len(raw)) + raw


# ---------------------------------------------------------------------------
# XML builders
# ---------------------------------------------------------------------------

def _vtr_xml(nx, ny, nz, ft, o_mag, o_vec, o_xc, o_yc, o_zc) -> bytes:
    s = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="0.1"'
        ' byte_order="LittleEndian" header_type="UInt64">\n'
        f'  <RectilinearGrid WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n'
        f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n'
        '      <PointData>\n'
        f'        <DataArray type="Float32" Name="{ft}_magnitude"'
        f' format="appended" offset="{o_mag}"/>\n'
        f'        <DataArray type="Float32" Name="{ft}_vector"'
        f' NumberOfComponents="3" format="appended" offset="{o_vec}"/>\n'
        '      </PointData>\n'
        '      <Coordinates>\n'
        f'        <DataArray type="Float32" Name="x"'
        f' format="appended" offset="{o_xc}"/>\n'
        f'        <DataArray type="Float32" Name="y"'
        f' format="appended" offset="{o_yc}"/>\n'
        f'        <DataArray type="Float32" Name="z"'
        f' format="appended" offset="{o_zc}"/>\n'
        '      </Coordinates>\n'
        '    </Piece>\n'
        '  </RectilinearGrid>\n'
        '  <AppendedData encoding="raw">\n_'
    )
    return s.encode("ascii")


def _vtr_offsets(nx, ny, nz):
    H = 8  # UInt64 header = 8 bytes
    n = nx * ny * nz
    o_mag = 0
    o_vec = o_mag + H + n * 4
    o_xc  = o_vec + H + n * 3 * 4
    o_yc  = o_xc  + H + nx * 4
    o_zc  = o_yc  + H + ny * 4
    return o_mag, o_vec, o_xc, o_yc, o_zc


# ---------------------------------------------------------------------------
# Structured .vtr  (in-memory)
# ---------------------------------------------------------------------------

def write_vtr(result, output_path):
    """
    Write a StructuredResult to a binary-appended RectilinearGrid (.vtr).
    Loads all data into memory — use write_vtr_streaming() for large files.
    """
    r = result
    nx, ny, nz = len(r.x), len(r.y), len(r.z)

    E_mag = np.sqrt(
        np.abs(r.Ex) ** 2 + np.abs(r.Ey) ** 2 + np.abs(r.Ez) ** 2
    ).astype(np.float32)

    Ex_r = np.real(r.Ex).astype(np.float32)
    Ey_r = np.real(r.Ey).astype(np.float32)
    Ez_r = np.real(r.Ez).astype(np.float32)

    # CST (nz,ny,nx) → VTK Fortran order (x varies fastest)
    def vtk_order(a):
        return a.T.flatten(order="F")

    mag_flat = vtk_order(E_mag)
    ex_flat  = vtk_order(Ex_r)
    ey_flat  = vtk_order(Ey_r)
    ez_flat  = vtk_order(Ez_r)

    n = nx * ny * nz
    vec_flat = np.empty(n * 3, dtype=np.float32)
    vec_flat[0::3] = ex_flat
    vec_flat[1::3] = ey_flat
    vec_flat[2::3] = ez_flat

    o_mag, o_vec, o_xc, o_yc, o_zc = _vtr_offsets(nx, ny, nz)

    with open(output_path, "wb") as f:
        f.write(_vtr_xml(nx, ny, nz, r.field_type, o_mag, o_vec, o_xc, o_yc, o_zc))
        f.write(_block(mag_flat))
        f.write(_block(vec_flat))
        f.write(_block(r.x))
        f.write(_block(r.y))
        f.write(_block(r.z))
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")


# ---------------------------------------------------------------------------
# Structured .vtr  (streaming — safe for large files)
# ---------------------------------------------------------------------------

def write_vtr_streaming(filepath, output_path, subsample=1,
                        slice_axis=None, slice_idx=0, time_idx=None):
    """
    Convert a structured CST HDF5 file to .vtr without loading it into RAM.

    Reads the E/H-Field in z-slabs (one at a time).  Magnitude is written
    directly to the output file; vector data is spooled to a temp file and
    appended afterwards — so each slab is read only once.

    Parameters
    ----------
    filepath   : path to CST HDF5 file
    output_path: destination .vtr file
    subsample  : keep every Nth point in all axes
    slice_axis : 'x', 'y', 'z', or None (full 3-D)
    slice_idx  : grid index for the fixed axis when slice_axis is set
    time_idx   : for structured_time (nT, nz, ny, nx) files — which time step
                 to stream; ignored for plain structured (nz, ny, nx) files
    """
    import h5py
    from .reader import _field_key, _build_selection

    with h5py.File(filepath, "r") as f:
        fkey = _field_key(f)
        ft = "E" if "E-Field" in fkey else "H"
        x_all = f["Mesh line x"][:].astype(np.float32)
        y_all = f["Mesh line y"][:].astype(np.float32)
        z_all = f["Mesh line z"][:].astype(np.float32)
        # Use spatial shape only — works for both (nz,ny,nx) and (nT,nz,ny,nx)
        shape_full = f[fkey].shape[-3:]
        ti = 0 if time_idx is None else time_idx

        _, x, y, z, iz_list, iy_sel, ix_sel = _build_selection(
            shape_full, subsample, slice_axis, slice_idx, x_all, y_all, z_all
        )
        nx, ny, nz = len(x), len(y), len(z)
        n_pts = nx * ny * nz
        mag_nbytes = n_pts * 4
        vec_nbytes = n_pts * 3 * 4
        _xdt = f[fkey].dtype["x"]
        is_complex = _xdt.names is not None and "re" in _xdt.names
        is_4d = f[fkey].ndim == 4

        o_mag, o_vec, o_xc, o_yc, o_zc = _vtr_offsets(nx, ny, nz)

        out = open(output_path, "wb")
        out.write(_vtr_xml(nx, ny, nz, ft, o_mag, o_vec, o_xc, o_yc, o_zc))
        out.write(_hdr(mag_nbytes))

        with tempfile.TemporaryFile() as vec_tmp:
            vec_tmp.write(_hdr(vec_nbytes))

            for iz in iz_list:
                if is_4d:
                    slab = f[fkey][ti, iz, iy_sel, ix_sel]  # (ny', nx')
                else:
                    slab = f[fkey][iz, iy_sel, ix_sel]      # (ny', nx')

                if is_complex:
                    Fxs = (slab["x"]["re"] + 1j * slab["x"]["im"]).astype(np.complex64)
                    Fys = (slab["y"]["re"] + 1j * slab["y"]["im"]).astype(np.complex64)
                    Fzs = (slab["z"]["re"] + 1j * slab["z"]["im"]).astype(np.complex64)
                    mag = np.sqrt(
                        np.abs(Fxs) ** 2 + np.abs(Fys) ** 2 + np.abs(Fzs) ** 2
                    ).astype(np.float32)
                    fx_r = Fxs.real.astype(np.float32)
                    fy_r = Fys.real.astype(np.float32)
                    fz_r = Fzs.real.astype(np.float32)
                else:
                    fx_r = slab["x"].astype(np.float32)
                    fy_r = slab["y"].astype(np.float32)
                    fz_r = slab["z"].astype(np.float32)
                    mag = np.sqrt(fx_r ** 2 + fy_r ** 2 + fz_r ** 2)

                # C-order flatten: for slab (ny', nx'), x varies fastest ✓
                out.write(mag.flatten(order="C").tobytes())

                ns = mag.size
                vec = np.empty(ns * 3, dtype=np.float32)
                vec[0::3] = fx_r.flatten(order="C")
                vec[1::3] = fy_r.flatten(order="C")
                vec[2::3] = fz_r.flatten(order="C")
                vec_tmp.write(vec.tobytes())

            # Append vector block from temp file
            vec_tmp.seek(0)
            chunk = 4 * 1024 * 1024  # 4 MB
            while True:
                buf = vec_tmp.read(chunk)
                if not buf:
                    break
                out.write(buf)

        # Coordinate arrays
        for arr in (x, y, z):
            out.write(_block(arr))

        out.write(b"\n  </AppendedData>\n</VTKFile>\n")
        out.close()


# ---------------------------------------------------------------------------
# Unstructured .vtu  (point cloud)
# ---------------------------------------------------------------------------

def write_vtu(result, output_path):
    """Write an UnstructuredResult to a binary-appended .vtu (VTK_VERTEX cells)."""
    r = result
    N = r.positions.shape[0]

    F_mag = np.sqrt(
        np.abs(r.Fx) ** 2 + np.abs(r.Fy) ** 2 + np.abs(r.Fz) ** 2
    ).astype(np.float32)
    Fx_r = np.real(r.Fx).astype(np.float32)
    Fy_r = np.real(r.Fy).astype(np.float32)
    Fz_r = np.real(r.Fz).astype(np.float32)

    vec = np.column_stack([Fx_r, Fy_r, Fz_r]).flatten().astype(np.float32)
    pts = r.positions.flatten().astype(np.float32)
    conn = np.arange(N, dtype=np.int32)
    offs = np.arange(1, N + 1, dtype=np.int32)
    types = np.ones(N, dtype=np.uint8)

    arrays = [
        (F_mag, np.float32),   # magnitude
        (vec,   np.float32),   # vector
        (pts,   np.float32),   # point coordinates
        (conn,  np.int32),     # connectivity
        (offs,  np.int32),     # offsets
        (types, np.uint8),     # cell types
    ]
    H = 8
    offsets_xml = []
    raw_blocks = []
    off = 0
    for arr, dt in arrays:
        raw = arr.astype(dt).tobytes()
        raw_blocks.append(_hdr(len(raw)) + raw)
        offsets_xml.append(off)
        off += H + len(raw)

    o_mag, o_vec, o_pts, o_conn, o_offsets, o_types = offsets_xml
    ft = r.field_type

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1"'
        ' byte_order="LittleEndian" header_type="UInt64">\n'
        "  <UnstructuredGrid>\n"
        f'    <Piece NumberOfPoints="{N}" NumberOfCells="{N}">\n'
        "      <PointData>\n"
        f'        <DataArray type="Float32" Name="{ft}_magnitude"'
        f' format="appended" offset="{o_mag}"/>\n'
        f'        <DataArray type="Float32" Name="{ft}_vector"'
        f' NumberOfComponents="3" format="appended" offset="{o_vec}"/>\n'
        "      </PointData>\n"
        "      <Points>\n"
        f'        <DataArray type="Float32" NumberOfComponents="3"'
        f' format="appended" offset="{o_pts}"/>\n'
        "      </Points>\n"
        "      <Cells>\n"
        f'        <DataArray type="Int32" Name="connectivity"'
        f' format="appended" offset="{o_conn}"/>\n'
        f'        <DataArray type="Int32" Name="offsets"'
        f' format="appended" offset="{o_offsets}"/>\n'
        f'        <DataArray type="UInt8" Name="types"'
        f' format="appended" offset="{o_types}"/>\n'
        "      </Cells>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "  <AppendedData encoding=\"raw\">\n_"
    )

    with open(output_path, "wb") as f:
        f.write(xml.encode("ascii"))
        for block in raw_blocks:
            f.write(block)
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")


# ---------------------------------------------------------------------------
# PVD collection
# ---------------------------------------------------------------------------

def write_pvd(entries, output_path):
    """
    Write a ParaView Data Collection file (.pvd) for multi-step animation.

    Parameters
    ----------
    entries     : list of (timestep_value, vtk_filepath_relative_to_pvd) tuples
    output_path : destination .pvd file
    """
    lines = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="Collection" version="0.1">\n',
        "  <Collection>\n",
    ]
    for t, fpath in entries:
        lines.append(
            f'    <DataSet timestep="{t}" group="" part="0" file="{fpath}"/>\n'
        )
    lines += ["  </Collection>\n", "</VTKFile>\n"]
    with open(output_path, "w") as f:
        f.writelines(lines)
