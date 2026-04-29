"""
CST Studio Field Results -> ParaView Visualization
===================================================
Edit the CONFIGURATION section, then scroll to the bottom
and uncomment the function call(s) you want to run.

Supported HDF5 formats (auto-detected):
  structured       - 3D rectilinear grid, FIT/FEM frequency-domain snapshot
  structured_time  - 3D rectilinear grid with time axis, FIT time-domain monitor
  unstructured     - point cloud at arbitrary positions, open-boundary FEM
  time_animation   - unstructured time-domain monitor (multiple frames)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from paraview_pipeline.reader import (
    detect_format,
    read_structured,
    read_unstructured,
    get_time_steps,
    get_structured_shape,
)
from paraview_pipeline.plot import plot_structured_slice, plot_unstructured
from paraview_pipeline.convert import (
    convert_structured,
    convert_unstructured,
    convert_time_series,
    convert_structured_time_series,
    convert_batch,
    convert_phase_animation,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  <- Edit these paths and parameters
# ─────────────────────────────────────────────────────────────────────────────

# Folder containing your CST HDF5 field result files
# INPUT_DIR = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\Field dump in HDF5"
INPUT_DIR = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\dielectric_filled_waveguide\Export\3d"

# Where VTK output files will be written
OUTPUT_DIR = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\dielectric_filled_waveguide\vtk_output"

# ── Single-file operations ───────────────────────────────────────────────────
# Path to the specific .h5 file you want to convert or plot
INPUT_FILE = os.path.join(INPUT_DIR, "e-field (f=67) [1].h5")

# ── Resolution / memory controls ────────────────────────────────────────────
# Keep every Nth point in each axis.
#   1  = full resolution (may require many GB of RAM for large files)
#   4  = good starting point for 20 GB files (~300x smaller in RAM)
#   8  = very fast preview
SUBSAMPLE = 1

# ── 2D slice extraction ──────────────────────────────────────────────────────
# Extract a single 2D plane instead of the full volume.
# Set SLICE_AXIS to None to convert the full 3D volume.
SLICE_AXIS = "z"    # 'x', 'y', 'z', or None
SLICE_IDX  = 7      # grid index along SLICE_AXIS (use inspect_files() for valid range)

# ── Plot options ─────────────────────────────────────────────────────────────
# Which quantity to colour-map:
#   'magnitude'                     - |E| or |H|
#   'x', 'y', 'z'                   - component magnitudes
#   'phase_x', 'phase_y', 'phase_z' - phase angle in degrees (freq-domain only)
PLOT_COMPONENT = "magnitude"
LOG_SCALE      = True   # log colour scale (recommended for E/H fields)
PLOT_AXIS      = "z"    # axis perpendicular to the plotted slice
PLOT_IDX       = 7      # index of the slice to plot (None = middle of domain)
SAVE_PLOT      = True   # if True, saves a PNG alongside the VTK output

# ── Streaming mode (for files that don't fit in RAM) ─────────────────────────
# write_vtr_streaming() reads one z-plane at a time; auto-enabled when the
# estimated uncompressed output exceeds 1 GB.  Force it explicitly here:
FORCE_STREAMING = False

# ── Force in-memory mode (override auto-streaming for large files) ────────────
# Set True if you have >= 64 GB of RAM and want the fastest conversion path.
# With 128 GB RAM this is safe even for files that would otherwise auto-stream.
FORCE_MEMORY = False

# ── Phase animation options ───────────────────────────────────────────────────
# Number of phasor angle steps from 0 to 360 degrees.
#   36  = 10 deg/frame  (smooth, standard)
#   72  = 5 deg/frame   (very smooth, 2x more .vtr files)
#   18  = 20 deg/frame  (quick preview)
N_PHASE_FRAMES = 36
# Try to use the NVIDIA GPU (CuPy) for the phasor multiply-accumulate loop.
# Falls back to NumPy automatically if CuPy is not installed or field doesn't
# fit in VRAM (A4500 has 16 GB; NumPy is used for larger datasets).
USE_GPU = True

# ── Batch conversion options ─────────────────────────────────────────────────
BATCH_PATTERN = "*.h5"   # glob pattern for scanning INPUT_DIR
WRITE_PVD     = True     # write a .pvd animation index alongside .vtu series

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 0 — Inspect files in INPUT_DIR
# ─────────────────────────────────────────────────────────────────────────────

def inspect_files():
    """
    Print the HDF5 format, array shape, and file size for every .h5 in INPUT_DIR.
    Run this first to find valid SLICE_IDX ranges for your files.
    """
    import h5py

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".h5"))
    if not files:
        print("No .h5 files found in " + INPUT_DIR)
        return

    print(f"\n{'File':<55} {'Format':<16} {'Shape':<22} {'MB':>6}")
    print("-" * 105)
    for fn in files:
        path = os.path.join(INPUT_DIR, fn)
        size_mb = os.path.getsize(path) / 1e6
        fmt = detect_format(path)
        if fmt == "structured":
            nz, ny, nx = get_structured_shape(path)
            shape_str = f"({nz}, {ny}, {nx}) [z,y,x]"
            extra = (
                f"  valid SLICE_IDX: z=0..{nz-1}  y=0..{ny-1}  x=0..{nx-1}"
            )
        elif fmt == "structured_time":
            times = get_time_steps(path)
            nz, ny, nx = get_structured_shape(path)
            nt = len(times) if times is not None else "?"
            shape_str = f"({nt}, {nz}, {ny}, {nx}) [t,z,y,x]"
            extra = (
                f"  {nt} steps  t={times[0]:.3f}..{times[-1]:.3f}"
                if times is not None else ""
            )
        else:
            times = get_time_steps(path)
            with h5py.File(path, "r") as hf:
                fkey = next(k for k in hf.keys() if "Field" in k)
                shape_str = str(hf[fkey].shape)
            extra = (
                f"  n_time_steps={len(times)}" if times is not None else ""
            )
        print(f"  {fn:<53} {fmt:<16} {shape_str:<22} {size_mb:>6.1f}{extra}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 1 — Convert a single structured file (full 3D volume)
# ─────────────────────────────────────────────────────────────────────────────

def convert_full_volume():
    """
    Convert INPUT_FILE to a .vtr RectilinearGrid.
    If SUBSAMPLE > 1 or FORCE_STREAMING=True the streaming path is used.

    Output: <OUTPUT_DIR>/<stem>.vtr
    """
    stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    tag  = f"_sub{SUBSAMPLE}" if SUBSAMPLE > 1 else ""
    out  = os.path.join(OUTPUT_DIR, f"{stem}{tag}.vtr")

    fmt = detect_format(INPUT_FILE)
    if fmt != "structured":
        print(
            f"ERROR: {os.path.basename(INPUT_FILE)} is '{fmt}', "
            "not structured. Use convert_unstructured_file() instead."
        )
        return

    convert_structured(INPUT_FILE, out, subsample=SUBSAMPLE,
                       streaming=FORCE_STREAMING,
                       force_memory=FORCE_MEMORY)
    print(f"Done.  Open in ParaView: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 2 — Extract a 2D slice from a structured file
# ─────────────────────────────────────────────────────────────────────────────

def convert_slice():
    """
    Extract a single 2D cross-section from INPUT_FILE.
    Uses SLICE_AXIS and SLICE_IDX from the configuration above.

    Output: <OUTPUT_DIR>/<stem>_slice_<axis><idx>.vtr
    """
    stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    out  = os.path.join(
        OUTPUT_DIR, f"{stem}_slice_{SLICE_AXIS}{SLICE_IDX}.vtr"
    )

    fmt = detect_format(INPUT_FILE)
    if fmt != "structured":
        print(f"ERROR: {os.path.basename(INPUT_FILE)} is '{fmt}', not structured.")
        return

    convert_structured(INPUT_FILE, out,
                       subsample=SUBSAMPLE,
                       slice_axis=SLICE_AXIS,
                       slice_idx=SLICE_IDX)
    print(f"Done.  Open in ParaView: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 3 — Convert a single unstructured / time-animation file
# ─────────────────────────────────────────────────────────────────────────────

def convert_unstructured_file(time_idx=0):
    """
    Convert a point-cloud or time-animation HDF5 to .vtu.

    For time-animation files this converts a single frame (time_idx=0 by default).
    To convert all frames at once, use convert_all_time_steps() below.

    Output: <OUTPUT_DIR>/<stem>_t<time_idx>.vtu  (or <stem>.vtu if not time-varying)
    """
    stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    fmt  = detect_format(INPUT_FILE)

    if fmt == "time_animation":
        out = os.path.join(OUTPUT_DIR, f"{stem}_t{time_idx:04d}.vtu")
    else:
        out = os.path.join(OUTPUT_DIR, f"{stem}.vtu")

    convert_unstructured(INPUT_FILE, out, time_idx=time_idx)
    print(f"Done.  Open in ParaView: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 4 — Convert all time steps (time-animation files)
# ─────────────────────────────────────────────────────────────────────────────

def convert_all_time_steps():
    """
    Convert every time frame in INPUT_FILE to a separate .vtu.
    Writes one .pvd collection file so ParaView can animate them
    (File -> Open the .pvd file, then press Play).

    Output folder: <OUTPUT_DIR>/<stem>/
    Output .pvd:   <OUTPUT_DIR>/<stem>/<stem>.pvd
    """
    stem    = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    out_dir = os.path.join(OUTPUT_DIR, stem)

    times = get_time_steps(INPUT_FILE)
    if times is None:
        print(
            f"ERROR: {os.path.basename(INPUT_FILE)} has no Times dataset. "
            "It may be a frequency-domain file."
        )
        return

    print(f"Converting {len(times)} time steps from {os.path.basename(INPUT_FILE)}")
    convert_time_series(INPUT_FILE, out_dir, write_pvd_file=WRITE_PVD)
    pvd = os.path.join(out_dir, stem + ".pvd")
    print(f"Done.  Load animation in ParaView: {pvd}")


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 5 — Batch-convert all .h5 files in INPUT_DIR
# ─────────────────────────────────────────────────────────────────────────────

def batch_convert_all():
    """
    Convert every matching HDF5 file in INPUT_DIR.

    - Structured files   -> individual .vtr + one .pvd per field type
    - Unstructured files -> individual .vtu + one .pvd per field type
    - Time-animation     -> per-file subfolder with .vtu series + .pvd

    Output root: <OUTPUT_DIR>/batch/
    """
    out_dir = os.path.join(OUTPUT_DIR, "batch")
    print(f"Batch converting: {INPUT_DIR}")
    print(f"         pattern: {BATCH_PATTERN}")
    print(f"       output to: {out_dir}\n")
    convert_batch(INPUT_DIR, out_dir,
                  pattern=BATCH_PATTERN,
                  subsample=SUBSAMPLE,
                  write_pvd_file=WRITE_PVD,
                  streaming=FORCE_STREAMING)
    print(f"\nDone.  Output folder: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 6 — Plot a 2D field contour with matplotlib (no ParaView needed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_field():
    """
    Show a 2D colour-map + quiver-direction overlay of INPUT_FILE using matplotlib.

    For structured files: plots the slice at (PLOT_AXIS, PLOT_IDX).
    For unstructured files: scatter plot projected onto x-y and x-z planes.

    If SAVE_PLOT=True, also saves a PNG to OUTPUT_DIR.
    """
    stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    fmt  = detect_format(INPUT_FILE)

    if fmt == "structured":
        png = (
            os.path.join(
                OUTPUT_DIR,
                f"{stem}_{PLOT_AXIS}{PLOT_IDX}_{PLOT_COMPONENT}.png",
            )
            if SAVE_PLOT else None
        )
        result = read_structured(
            INPUT_FILE,
            subsample=SUBSAMPLE,
            slice_axis=PLOT_AXIS if PLOT_IDX is not None else None,
            slice_idx=PLOT_IDX or 0,
        )
        plot_structured_slice(result,
                              axis=PLOT_AXIS,
                              idx=PLOT_IDX,
                              component=PLOT_COMPONENT,
                              log_scale=LOG_SCALE,
                              output_path=png,
                              show=True)
    else:
        png = (
            os.path.join(OUTPUT_DIR, f"{stem}_{PLOT_COMPONENT}.png")
            if SAVE_PLOT else None
        )
        result = read_unstructured(INPUT_FILE)
        plot_unstructured(result,
                          component=PLOT_COMPONENT,
                          log_scale=LOG_SCALE,
                          output_path=png,
                          show=True)


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 7 — Plot all three axis slices at once (structured only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_slices():
    """
    Read INPUT_FILE once, then plot x, y, and z mid-plane slices.
    Useful for a quick 3D overview without opening ParaView.

    Saves three PNGs to OUTPUT_DIR if SAVE_PLOT=True.
    """
    fmt = detect_format(INPUT_FILE)
    if fmt != "structured":
        print("ERROR: plot_three_slices() only works on structured files.")
        return

    result = read_structured(INPUT_FILE, subsample=max(1, SUBSAMPLE))
    stem   = os.path.splitext(os.path.basename(INPUT_FILE))[0]

    for ax in ("x", "y", "z"):
        png = (
            os.path.join(OUTPUT_DIR, f"{stem}_{ax}mid_{PLOT_COMPONENT}.png")
            if SAVE_PLOT else None
        )
        plot_structured_slice(result,
                              axis=ax,
                              idx=None,
                              component=PLOT_COMPONENT,
                              log_scale=LOG_SCALE,
                              output_path=png,
                              show=True)


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 8 — Phase animation (frequency-domain phasor sweep)
# ─────────────────────────────────────────────────────────────────────────────

def phase_animation():
    """
    Sweep the phasor angle from 0 to 360 degrees and write one .vtr per frame.
    INPUT_FILE must be a structured frequency-domain (complex) HDF5 file.

    The result is a .pvd collection you can open in ParaView and play like a
    movie to see how the instantaneous field rotates through the RF cycle.

    GPU acceleration is used automatically when CuPy is installed and the
    field fits in VRAM; otherwise falls back to NumPy (still fast with 128 GB).

    Output folder: <OUTPUT_DIR>/<stem>_phase/
    Output .pvd:   <OUTPUT_DIR>/<stem>_phase/<stem>_phase_animation.pvd
    """
    stem    = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    out_dir = os.path.join(OUTPUT_DIR, f"{stem}_phase")

    print(f"Phase animation: {os.path.basename(INPUT_FILE)}")
    print(f"  frames   : {N_PHASE_FRAMES}  ({360 // N_PHASE_FRAMES} deg/frame)")
    print(f"  subsample: {SUBSAMPLE}")
    print(f"  GPU      : {USE_GPU}")
    print(f"  output   : {out_dir}\n")

    pvd = convert_phase_animation(
        INPUT_FILE, out_dir,
        n_frames=N_PHASE_FRAMES,
        subsample=SUBSAMPLE,
        write_pvd_file=WRITE_PVD,
        use_gpu=USE_GPU,
    )
    if pvd:
        print(f"\nDone.  Load animation in ParaView: {pvd}")


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 9 — Convert all time steps of a structured time-domain file
# ─────────────────────────────────────────────────────────────────────────────

def convert_structured_time_steps():
    """
    Convert every time step in INPUT_FILE (a structured_time HDF5) to
    individual .vtr RectilinearGrid files and a .pvd collection.

    INPUT_FILE must have format 'structured_time' — i.e. an E/H-field monitor
    exported from a FIT time-domain simulation (shape: nT x nz x ny x nx).
    Use inspect_files() to confirm.

    Open the .pvd file in ParaView, then press Play to animate the field
    evolving over the simulation time window.

    Output folder: <OUTPUT_DIR>/<stem>/
    Output .pvd:   <OUTPUT_DIR>/<stem>/<stem>.pvd
    """
    fmt = detect_format(INPUT_FILE)
    if fmt != "structured_time":
        print(
            f"ERROR: {os.path.basename(INPUT_FILE)} is '{fmt}', "
            "not structured_time. "
            "For unstructured time-animation use convert_all_time_steps()."
        )
        return

    stem    = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    out_dir = os.path.join(OUTPUT_DIR, stem)

    times = get_time_steps(INPUT_FILE)
    print(f"Converting {len(times)} time steps from {os.path.basename(INPUT_FILE)}")
    print(f"  t = {times[0]:.4f} .. {times[-1]:.4f}")
    print(f"  subsample : {SUBSAMPLE}")
    print(f"  output    : {out_dir}\n")

    convert_structured_time_series(
        INPUT_FILE, out_dir,
        subsample=SUBSAMPLE,
        write_pvd_file=WRITE_PVD,
        streaming=FORCE_STREAMING,
        force_memory=FORCE_MEMORY,
    )
    pvd = os.path.join(out_dir, stem + ".pvd")
    if WRITE_PVD:
        print(f"\nDone.  Load animation in ParaView: {pvd}")
    else:
        print(f"\nDone.  Output folder: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN  <- Uncomment the operation(s) you want to execute
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 0: print what files exist and their shapes/sizes
    inspect_files()

    # Step 1: convert the full 3D volume to .vtr (use SUBSAMPLE=4+ for large files)
    # convert_full_volume()

    # Step 2: extract one 2D slice (fastest option, minimal RAM)
    # convert_slice()

    # Step 3: convert a single unstructured snapshot or one time frame
    # convert_unstructured_file(time_idx=0)

    # Step 4: convert all time steps -> .pvd animation for ParaView
    # convert_all_time_steps()

    # Step 5: batch-convert every .h5 in INPUT_DIR
    # batch_convert_all()

    # Step 6: plot a single 2D slice with matplotlib (no ParaView needed)
    # plot_field()

    # Step 7: plot mid-plane slices along all three axes
    plot_three_slices()

    # Step 8: phasor-sweep animation (frequency-domain files only)
    # phase_animation()

    # Step 9: convert all time steps of a structured time-domain file -> .pvd animation
    # convert_structured_time_steps()
