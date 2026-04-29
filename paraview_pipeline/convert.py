"""
Batch converter and CLI for CST HDF5 → ParaView VTK.

Single-file usage
-----------------
  python paraview_pipeline/convert.py structured  INPUT.h5  OUTPUT.vtr
      [--subsample N] [--slice-axis z|y|x] [--slice-idx 0] [--stream]
      [--force-memory]

  python paraview_pipeline/convert.py unstructured INPUT.h5  OUTPUT.vtu
      [--time-idx T]

  python paraview_pipeline/convert.py time-series  INPUT.h5  OUTPUT_DIR/
      [--no-pvd]       (unstructured time-animation: one .vtu per step)

  python paraview_pipeline/convert.py structured-time-series INPUT.h5 OUTPUT_DIR/
      [--subsample N] [--no-pvd] [--stream] [--force-memory]
                       (structured time-domain: one .vtr per step)

  python paraview_pipeline/convert.py phase-animation INPUT.h5 OUTPUT_DIR/
      [--frames N] [--subsample N] [--no-pvd] [--no-gpu]

Batch (frequency sweep) usage
------------------------------
  python paraview_pipeline/convert.py batch  INPUT_DIR/  OUTPUT_DIR/
      [--pattern "*.h5"] [--subsample N] [--pvd] [--stream]

Plot usage
----------
  python paraview_pipeline/convert.py plot INPUT.h5
      [--slice-axis z] [--slice-idx 0] [--component magnitude]
      [--log] [--output fig.png] [--no-show]

Python API
----------
  from paraview_pipeline.convert import (
      convert_structured, convert_structured_time_series,
      convert_batch, convert_phase_animation,
  )
"""
import argparse
import os
import re
import sys
from pathlib import Path

# Allow `python paraview_pipeline/convert.py <cmd>` in addition to
# `python -m paraview_pipeline.convert <cmd>`
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "paraview_pipeline"  # enables relative imports in functions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freq_from_name(name: str):
    """Extract frequency in Hz from filenames like 'E-field67GHz.h5'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GHz|MHz|kHz|Hz)", name, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        mult = {"GHZ": 1e9, "MHZ": 1e6, "KHZ": 1e3, "HZ": 1.0}[unit]
        return val * mult
    return None


def _stem(filepath: str) -> str:
    return Path(filepath).stem


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def convert_structured(input_h5, output_vtr, subsample=1,
                       slice_axis=None, slice_idx=0, streaming=False,
                       force_memory=False):
    """
    Convert a single structured HDF5 file to .vtr.

    streaming=True    : force z-slab streaming (low RAM, slightly slower)
    force_memory=True : force in-memory path even for large files
                        (safe when you have plenty of RAM, e.g. 128 GB)
    Auto-streaming is enabled when slice_axis=None, subsample=1, and the
    estimated output exceeds 1 GB — unless force_memory=True overrides it.
    """
    from .reader import read_structured, get_structured_shape
    from .writer import write_vtr, write_vtr_streaming

    shape = get_structured_shape(input_h5)
    nz, ny, nx = shape
    n_pts = (nz // subsample or 1) * (ny // subsample or 1) * (nx // subsample or 1)
    est_gb = n_pts * 4 / 1e9  # rough RAM estimate (float32 magnitude only)

    use_stream = (not force_memory) and (
        streaming or (slice_axis is None and subsample == 1 and est_gb > 1.0)
    )

    if use_stream:
        print(f"  [stream] {Path(input_h5).name}  shape={shape}  est={est_gb:.1f} GB")
        write_vtr_streaming(input_h5, output_vtr, subsample=subsample,
                            slice_axis=slice_axis, slice_idx=slice_idx)
    else:
        print(f"  [mem]    {Path(input_h5).name}  shape={shape}  est={est_gb:.1f} GB")
        result = read_structured(input_h5, subsample=subsample,
                                 slice_axis=slice_axis, slice_idx=slice_idx)
        write_vtr(result, output_vtr)

    print(f"           -> {output_vtr}")


def convert_unstructured(input_h5, output_vtu, time_idx=None):
    """Convert a single unstructured / time-animation HDF5 snapshot to .vtu."""
    from .reader import read_unstructured, detect_format
    from .writer import write_vtu

    fmt = detect_format(input_h5)
    if fmt == "structured":
        raise ValueError(f"{input_h5} is a structured file; use convert_structured()")

    result = read_unstructured(input_h5, time_idx=time_idx)
    write_vtu(result, output_vtu)
    print(f"  {Path(input_h5).name}  N={result.positions.shape[0]}  -> {output_vtu}")


def convert_time_series(input_h5, output_dir, write_pvd_file=True):
    """
    Convert every time step in a time-animation HDF5 to individual .vtu files.
    Optionally writes a .pvd collection for ParaView animation.

    Returns list of output .vtu paths.
    """
    from .reader import get_time_steps, detect_format
    from .writer import write_pvd

    os.makedirs(output_dir, exist_ok=True)
    times = get_time_steps(input_h5)
    if times is None:
        raise ValueError(f"{input_h5} has no Times dataset")

    stem = _stem(input_h5)
    vtu_paths = []
    pvd_entries = []

    for ti, t in enumerate(times):
        out_name = f"{stem}_t{ti:04d}.vtu"
        out_path = os.path.join(output_dir, out_name)
        convert_unstructured(input_h5, out_path, time_idx=ti)
        vtu_paths.append(out_path)
        pvd_entries.append((float(t), out_name))

    if write_pvd_file:
        pvd_path = os.path.join(output_dir, f"{stem}.pvd")
        from .writer import write_pvd
        write_pvd(pvd_entries, pvd_path)
        print(f"PVD collection -> {pvd_path}")

    return vtu_paths


def convert_structured_time_series(input_h5, output_dir, subsample=1,
                                   write_pvd_file=True, streaming=False,
                                   force_memory=False):
    """
    Convert every time step in a structured_time HDF5 to individual .vtr files.
    Optionally writes a .pvd collection for ParaView animation.

    streaming=True / force_memory=True have the same meaning as in
    convert_structured().  Auto-streaming is enabled when subsample=1 and the
    estimated per-frame output exceeds 1 GB.

    Returns list of output .vtr paths.
    """
    from .reader import (detect_format, get_time_steps, get_structured_shape,
                         read_structured_time)
    from .writer import write_vtr, write_vtr_streaming, write_pvd

    fmt = detect_format(input_h5)
    if fmt != "structured_time":
        raise ValueError(
            f"{Path(input_h5).name} is '{fmt}'; expected structured_time. "
            "For unstructured time-animation use convert_time_series()."
        )

    os.makedirs(output_dir, exist_ok=True)
    times = get_time_steps(input_h5)
    if times is None:
        raise ValueError(f"{input_h5} has no Times dataset")

    shape = get_structured_shape(input_h5)
    nz, ny, nx = shape
    n_pts = (nz // subsample or 1) * (ny // subsample or 1) * (nx // subsample or 1)
    est_gb = n_pts * 4 / 1e9

    use_stream = (not force_memory) and (
        streaming or (subsample == 1 and est_gb > 1.0)
    )

    stem = _stem(input_h5)
    mode = "stream" if use_stream else "mem"
    print(f"  [{mode}] {Path(input_h5).name}  shape={shape}  "
          f"{len(times)} steps  est={est_gb:.2f} GB/frame")

    vtr_paths = []
    pvd_entries = []

    for ti, t in enumerate(times):
        out_name = f"{stem}_t{ti:04d}.vtr"
        out_path = os.path.join(output_dir, out_name)

        if use_stream:
            write_vtr_streaming(input_h5, out_path, subsample=subsample,
                                time_idx=ti)
        else:
            result = read_structured_time(input_h5, time_idx=ti,
                                          subsample=subsample)
            write_vtr(result, out_path)

        vtr_paths.append(out_path)
        pvd_entries.append((float(t), out_name))
        print(f"    step {ti+1:4d}/{len(times)}  t={t:.4f}  -> {out_name}")

    if write_pvd_file:
        pvd_path = os.path.join(output_dir, f"{stem}.pvd")
        write_pvd(pvd_entries, pvd_path)
        print(f"  PVD collection -> {pvd_path}")

    return vtr_paths


def convert_batch(input_dir, output_dir, pattern="*.h5", subsample=1,
                  write_pvd_file=True, streaming=False):
    """
    Batch-convert all matching HDF5 files in input_dir.

    Groups files by field type (E/H) and format (structured/unstructured/time).
    Writes one VTK file per HDF5 file plus an optional .pvd per group.

    Returns dict mapping input path → output path.
    """
    import glob
    from .reader import detect_format

    os.makedirs(output_dir, exist_ok=True)
    h5_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not h5_files:
        print(f"No files matching '{pattern}' in {input_dir}")
        return {}

    results = {}
    pvd_groups = {}   # group_key → list of (timestep, vtk_rel_path)

    for h5 in h5_files:
        fmt = detect_format(h5)
        stem = _stem(h5)
        freq = _freq_from_name(stem)

        if fmt == "structured":
            out = os.path.join(output_dir, stem + ".vtr")
            convert_structured(h5, out, subsample=subsample, streaming=streaming)
            results[h5] = out
            # Group by field type for PVD
            ft = "E" if "E-Field" in stem or "E-field" in stem else "H"
            key = f"{ft}_structured"
            pvd_groups.setdefault(key, []).append(
                (freq if freq is not None else len(pvd_groups.get(key, [])),
                 stem + ".vtr")
            )

        elif fmt == "structured_time":
            sub_dir = os.path.join(output_dir, stem)
            convert_structured_time_series(h5, sub_dir, subsample=subsample,
                                           write_pvd_file=write_pvd_file,
                                           streaming=streaming)
            results[h5] = sub_dir

        elif fmt == "time_animation":
            sub_dir = os.path.join(output_dir, stem)
            vtu_list = convert_time_series(h5, sub_dir, write_pvd_file=write_pvd_file)
            results[h5] = sub_dir

        else:  # unstructured
            out = os.path.join(output_dir, stem + ".vtu")
            convert_unstructured(h5, out)
            results[h5] = out
            ft = "E" if "E-Field" in stem or "E-field" in stem else "H"
            key = f"{ft}_unstructured"
            pvd_groups.setdefault(key, []).append(
                (freq if freq is not None else len(pvd_groups.get(key, [])),
                 stem + ".vtu")
            )

    # Write per-group PVD collections
    if write_pvd_file:
        from .writer import write_pvd
        for key, entries in pvd_groups.items():
            entries_sorted = sorted(entries, key=lambda e: e[0])
            pvd_path = os.path.join(output_dir, f"{key}.pvd")
            write_pvd(entries_sorted, pvd_path)
            print(f"PVD collection ({key}) -> {pvd_path}")

    return results


def convert_phase_animation(input_h5, output_dir, n_frames=36, subsample=1,
                             write_pvd_file=True, use_gpu=True):
    """
    Generate a phasor-sweep animation from a frequency-domain structured file.

    Computes E(theta) = Re(E_complex * exp(j*theta)) for theta in [0, 2*pi)
    and writes n_frames .vtr files plus a .pvd collection for ParaView.

    Raises ValueError if the file is not structured or not complex (i.e. it is
    a time-domain file, which already has its own time-series animation).

    Parameters
    ----------
    input_h5     : path to structured frequency-domain CST HDF5 file
    output_dir   : directory where .vtr frames and .pvd will be written
    n_frames     : number of phase steps (default 36 = 10 deg per frame)
    subsample    : keep every Nth point in all axes (useful for large files)
    write_pvd_file : write a .pvd animation collection (default True)
    use_gpu      : attempt CuPy GPU acceleration; falls back to NumPy silently

    Returns the .pvd path if write_pvd_file=True, else None.
    """
    import numpy as np
    from .reader import read_structured, detect_format, StructuredResult
    from .writer import write_vtr, write_pvd

    fmt = detect_format(input_h5)
    if fmt != "structured":
        raise ValueError(
            f"Phase animation requires a structured file; got '{fmt}'. "
            "For time-domain animations use convert_time_series()."
        )

    os.makedirs(output_dir, exist_ok=True)
    stem = _stem(input_h5)

    print(f"  Loading {Path(input_h5).name} ...")
    result = read_structured(input_h5, subsample=subsample)

    if not result.is_complex:
        raise ValueError(
            f"{Path(input_h5).name} contains real (time-domain) data. "
            "Phase animation requires complex (frequency-domain) data."
        )

    # --- choose compute backend ---
    xp = np
    Ex, Ey, Ez = result.Ex, result.Ey, result.Ez

    if use_gpu:
        try:
            import cupy as cp
            Ex = cp.asarray(result.Ex)
            Ey = cp.asarray(result.Ey)
            Ez = cp.asarray(result.Ez)
            xp = cp
            mem_gb = Ex.nbytes * 3 / 1e9
            print(f"  [GPU] CuPy: {mem_gb:.2f} GB on device")
        except Exception as e:
            Ex, Ey, Ez = result.Ex, result.Ey, result.Ez
            xp = np
            print(f"  [CPU] CuPy unavailable ({e}), using NumPy")

    # Pre-split real/imag once to avoid repeated views in the loop
    Ex_re, Ex_im = xp.real(Ex).astype(xp.float32), xp.imag(Ex).astype(xp.float32)
    Ey_re, Ey_im = xp.real(Ey).astype(xp.float32), xp.imag(Ey).astype(xp.float32)
    Ez_re, Ez_im = xp.real(Ez).astype(xp.float32), xp.imag(Ez).astype(xp.float32)

    angles = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    pvd_entries = []

    print(f"  Writing {n_frames} phase frames ...")
    for i, theta in enumerate(angles):
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        ex_f = Ex_re * c - Ex_im * s
        ey_f = Ey_re * c - Ey_im * s
        ez_f = Ez_re * c - Ez_im * s

        if xp is not np:
            import cupy as cp
            ex_f = cp.asnumpy(ex_f)
            ey_f = cp.asnumpy(ey_f)
            ez_f = cp.asnumpy(ez_f)

        frame = StructuredResult(
            x=result.x, y=result.y, z=result.z,
            Ex=ex_f, Ey=ey_f, Ez=ez_f,
            field_type=result.field_type,
            is_complex=False,
        )

        deg = round(np.degrees(theta))
        out_name = f"{stem}_phase{deg:03d}deg.vtr"
        out_path = os.path.join(output_dir, out_name)
        write_vtr(frame, out_path)
        pvd_entries.append((float(theta), out_name))
        print(f"    frame {i+1:3d}/{n_frames}  theta={deg:3d} deg -> {out_name}")

    pvd_path = None
    if write_pvd_file:
        pvd_path = os.path.join(output_dir, f"{stem}_phase_animation.pvd")
        write_pvd(pvd_entries, pvd_path)
        print(f"  PVD animation -> {pvd_path}")

    return pvd_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        prog="python -m paraview_pipeline.convert",
        description="Convert CST HDF5 field results to ParaView VTK files",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- structured ---------------------------------------------------------
    p_s = sub.add_parser("structured", help="Convert one structured HDF5 → .vtr")
    p_s.add_argument("input")
    p_s.add_argument("output")
    p_s.add_argument("--subsample", type=int, default=1, metavar="N",
                     help="Keep every Nth point in all axes")
    p_s.add_argument("--slice-axis", choices=["x", "y", "z"], default=None)
    p_s.add_argument("--slice-idx", type=int, default=0)
    p_s.add_argument("--stream", action="store_true",
                     help="Force z-slab streaming (auto-enabled for large files)")
    p_s.add_argument("--force-memory", action="store_true",
                     help="Force in-memory path even for large files (use with >=64 GB RAM)")

    # --- unstructured -------------------------------------------------------
    p_u = sub.add_parser("unstructured", help="Convert one unstructured HDF5 → .vtu")
    p_u.add_argument("input")
    p_u.add_argument("output")
    p_u.add_argument("--time-idx", type=int, default=None)

    # --- time-series (unstructured) -----------------------------------------
    p_t = sub.add_parser("time-series",
                         help="Convert all steps in a time-animation HDF5 → .vtu series")
    p_t.add_argument("input")
    p_t.add_argument("output_dir")
    p_t.add_argument("--no-pvd", action="store_true")

    # --- structured-time-series ---------------------------------------------
    p_st = sub.add_parser("structured-time-series",
                          help="Convert all steps in a structured_time HDF5 → .vtr series")
    p_st.add_argument("input")
    p_st.add_argument("output_dir")
    p_st.add_argument("--subsample", type=int, default=1, metavar="N")
    p_st.add_argument("--no-pvd", action="store_true")
    p_st.add_argument("--stream", action="store_true")
    p_st.add_argument("--force-memory", action="store_true")

    # --- batch --------------------------------------------------------------
    p_b = sub.add_parser("batch", help="Batch-convert all HDF5 in a directory")
    p_b.add_argument("input_dir")
    p_b.add_argument("output_dir")
    p_b.add_argument("--pattern", default="*.h5")
    p_b.add_argument("--subsample", type=int, default=1, metavar="N")
    p_b.add_argument("--no-pvd", action="store_true")
    p_b.add_argument("--stream", action="store_true")

    # --- phase-animation ----------------------------------------------------
    p_ph = sub.add_parser("phase-animation",
                           help="Phasor sweep animation from frequency-domain HDF5")
    p_ph.add_argument("input")
    p_ph.add_argument("output_dir")
    p_ph.add_argument("--frames", type=int, default=36, metavar="N",
                      help="Number of phase steps (default 36 = 10 deg each)")
    p_ph.add_argument("--subsample", type=int, default=1, metavar="N")
    p_ph.add_argument("--no-pvd", action="store_true")
    p_ph.add_argument("--no-gpu", action="store_true",
                      help="Disable CuPy GPU acceleration (use NumPy only)")

    # --- plot ---------------------------------------------------------------
    p_p = sub.add_parser("plot", help="Plot a 2-D slice with matplotlib")
    p_p.add_argument("input")
    p_p.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    p_p.add_argument("--slice-idx", type=int, default=None)
    p_p.add_argument("--component", default="magnitude",
                     choices=["magnitude", "x", "y", "z",
                               "phase_x", "phase_y", "phase_z"])
    p_p.add_argument("--subsample", type=int, default=1)
    p_p.add_argument("--log", action="store_true")
    p_p.add_argument("--output", default=None, metavar="IMG")
    p_p.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    if args.cmd == "structured":
        convert_structured(args.input, args.output,
                           subsample=args.subsample,
                           slice_axis=args.slice_axis,
                           slice_idx=args.slice_idx,
                           streaming=args.stream,
                           force_memory=args.force_memory)

    elif args.cmd == "unstructured":
        convert_unstructured(args.input, args.output, time_idx=args.time_idx)

    elif args.cmd == "time-series":
        convert_time_series(args.input, args.output_dir,
                            write_pvd_file=not args.no_pvd)

    elif args.cmd == "structured-time-series":
        convert_structured_time_series(args.input, args.output_dir,
                                       subsample=args.subsample,
                                       write_pvd_file=not args.no_pvd,
                                       streaming=args.stream,
                                       force_memory=args.force_memory)

    elif args.cmd == "batch":
        convert_batch(args.input_dir, args.output_dir,
                      pattern=args.pattern,
                      subsample=args.subsample,
                      write_pvd_file=not args.no_pvd,
                      streaming=args.stream)

    elif args.cmd == "phase-animation":
        convert_phase_animation(args.input, args.output_dir,
                                n_frames=args.frames,
                                subsample=args.subsample,
                                write_pvd_file=not args.no_pvd,
                                use_gpu=not args.no_gpu)

    elif args.cmd == "plot":
        from .reader import read_structured, detect_format
        from .plot import plot_structured_slice, plot_unstructured

        fmt = detect_format(args.input)
        if fmt == "structured":
            result = read_structured(args.input,
                                     subsample=args.subsample,
                                     slice_axis=args.slice_axis,
                                     slice_idx=args.slice_idx or 0)
            plot_structured_slice(result,
                                  axis=args.slice_axis or "z",
                                  idx=args.slice_idx,
                                  component=args.component,
                                  log_scale=args.log,
                                  output_path=args.output,
                                  show=not args.no_show)
        else:
            from .reader import read_unstructured
            result = read_unstructured(args.input)
            plot_unstructured(result,
                              component=args.component,
                              log_scale=args.log,
                              output_path=args.output,
                              show=not args.no_show)


if __name__ == "__main__":
    _cli()
