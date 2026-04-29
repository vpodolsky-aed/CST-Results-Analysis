"""
CST Studio Field Results -> Python / Matplotlib Visualization
=============================================================
Edit the CONFIGURATION section, then scroll to the bottom
and uncomment the function call(s) you want to run.

All operations stay inside Python — no ParaView or VTK required.
For VTK export and ParaView workflows, use main_to_paraview.py.

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
    read_structured_time,
    read_unstructured,
    get_time_steps,
    get_structured_shape,
)
from paraview_pipeline.plot import plot_structured_slice, plot_unstructured

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  <- Edit these paths and parameters
# ─────────────────────────────────────────────────────────────────────────────

# Folder containing your CST HDF5 field result files
# INPUT_DIR = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\Field dump in HDF5"
INPUT_DIR=r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\dielectric_filled_waveguide\Export_Parametric\0427-7669758\3d"

# Where PNG output files will be saved
OUTPUT_DIR = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\dielectric_filled_waveguide\Export_Parametric\0427-7669758\3d"

# ── Single-file operations ───────────────────────────────────────────────────
# Path to the specific .h5 file you want to process
# INPUT_FILE = os.path.join(INPUT_DIR, "Geometry1FIT67GHzE-field.h5")
INPUT_FILE = os.path.join(INPUT_DIR, "e-field (f=67) [1]_Pressure.h5")

# ── Resolution control ───────────────────────────────────────────────────────
# Keep every Nth point in each axis when loading into memory.
#   1  = full resolution
#   4  = good starting point for large files (~64x less RAM)
SUBSAMPLE = 1

# ── Plot options ─────────────────────────────────────────────────────────────
# Which quantity to colour-map:
#   'magnitude'                     - |E| or |H|
#   'x', 'y', 'z'                   - component magnitudes
#   'phase_x', 'phase_y', 'phase_z' - phase angle in degrees (freq-domain only)
PLOT_COMPONENT = "magnitude"
LOG_SCALE      = True   # log colour scale (recommended for E/H fields)
PLOT_AXIS      = "y"    # axis perpendicular to the plotted slice
PLOT_IDX       = None   # index of the slice to plot (None = middle of domain)
SAVE_PLOT      = True   # if True, saves a PNG to OUTPUT_DIR

# Colormap range — set to None to auto-scale from data
VMIN           = 1   # minimum value for colormap
VMAX           = None   # maximum value for colormap

# ── Time-domain frame selection ───────────────────────────────────────────────
# For structured_time or time_animation files: which time step to plot.
PLOT_TIME_IDX  = 0

# ── Animation options (plot_time_series only) ─────────────────────────────────
# "gif"  - animated GIF  (requires Pillow:  pip install Pillow)
# "mp4"  - H.264 video   (requires imageio + ffmpeg: pip install imageio[ffmpeg])
# None   - save PNGs only, no animation
ANIMATION_FORMAT   = "gif"
FRAME_DURATION_MS  = 100   # milliseconds each frame is displayed

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 0 — Inspect files in INPUT_DIR
# ─────────────────────────────────────────────────────────────────────────────

def inspect_files():
    """
    Print the HDF5 format, array shape, and file size for every .h5 in INPUT_DIR.
    Run this first to find valid SLICE_IDX / PLOT_IDX ranges for your files.
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
            extra = f"  valid PLOT_IDX: z=0..{nz-1}  y=0..{ny-1}  x=0..{nx-1}"
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
# OPERATION 1 — Plot a 2D field contour (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_field():
    """
    Show a 2D colour-map + quiver-direction overlay of INPUT_FILE using matplotlib.

    For structured files: plots the slice at (PLOT_AXIS, PLOT_IDX).
    For structured_time files: loads time step PLOT_TIME_IDX, same slice.
    For unstructured files: scatter plot projected onto x-y and x-z planes.

    If SAVE_PLOT=True, saves a PNG to OUTPUT_DIR.
    """
    stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    fmt  = detect_format(INPUT_FILE)

    if fmt in ("structured", "structured_time"):
        png = (
            os.path.join(
                OUTPUT_DIR,
                f"{stem}_{PLOT_AXIS}{PLOT_IDX}_{PLOT_COMPONENT}.png",
            )
            if SAVE_PLOT else None
        )
        if fmt == "structured_time":
            result = read_structured_time(
                INPUT_FILE,
                time_idx=PLOT_TIME_IDX,
                subsample=SUBSAMPLE,
                slice_axis=PLOT_AXIS if PLOT_IDX is not None else None,
                slice_idx=PLOT_IDX or 0,
            )
        else:
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
                              show=True,
                              vmin=VMIN,
                              vmax=VMAX,
                              title=stem)
    else:
        png = (
            os.path.join(OUTPUT_DIR, f"{stem}_{PLOT_COMPONENT}.png")
            if SAVE_PLOT else None
        )
        ti = PLOT_TIME_IDX if fmt == "time_animation" else None
        result = read_unstructured(INPUT_FILE, time_idx=ti)
        plot_unstructured(result,
                          component=PLOT_COMPONENT,
                          log_scale=LOG_SCALE,
                          output_path=png,
                          show=True,
                          vmin=VMIN,
                          vmax=VMAX,
                          title=stem)


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 2 — Plot all three axis slices at once (structured only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_slices():
    """
    Read INPUT_FILE once, then plot x, y, and z mid-plane slices.
    Useful for a quick 3D overview.

    For structured_time files, loads time step PLOT_TIME_IDX.
    Saves three PNGs to OUTPUT_DIR if SAVE_PLOT=True.
    """
    fmt = detect_format(INPUT_FILE)
    if fmt not in ("structured", "structured_time"):
        print("ERROR: plot_three_slices() only works on structured / structured_time files.")
        return

    stem = os.path.splitext(os.path.basename(INPUT_FILE))[0]

    if fmt == "structured_time":
        result = read_structured_time(INPUT_FILE, time_idx=PLOT_TIME_IDX,
                                      subsample=max(1, SUBSAMPLE))
    else:
        result = read_structured(INPUT_FILE, subsample=max(1, SUBSAMPLE))

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
                              show=True,
                              vmin=VMIN,
                              vmax=VMAX,
                              title=stem)


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 3 — Plot all time steps as a sequence of PNG files
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_series():
    """
    Loop over every time step in INPUT_FILE, display each frame briefly, and
    save one PNG per frame.  After all frames are written, assembles them into
    an animation whose format and per-frame duration are controlled by
    ANIMATION_FORMAT and FRAME_DURATION_MS.

    Works for both structured_time and time_animation formats.
    PNGs are saved to <OUTPUT_DIR>/<stem>_frames/.
    Animation is saved to <OUTPUT_DIR>/<stem>.<gif|mp4>.
    """
    import matplotlib.pyplot as plt

    fmt = detect_format(INPUT_FILE)
    if fmt not in ("structured_time", "time_animation"):
        print(f"ERROR: {os.path.basename(INPUT_FILE)} is '{fmt}', not a time-domain file.")
        return

    times = get_time_steps(INPUT_FILE)
    if times is None:
        print("ERROR: no Times dataset found in this file.")
        return

    stem    = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    out_dir = os.path.join(OUTPUT_DIR, f"{stem}_frames")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Plotting {len(times)} frames from {os.path.basename(INPUT_FILE)}")
    print(f"  t = {times[0]:.4e} .. {times[-1]:.4e}")
    print(f"  output: {out_dir}\n")

    plt.ion()
    png_paths = []

    for ti, t in enumerate(times):
        png = os.path.join(out_dir, f"{stem}_t{ti:04d}.png")
        png_paths.append(png)

        if fmt == "structured_time":
            result = read_structured_time(INPUT_FILE, time_idx=ti,
                                          subsample=SUBSAMPLE,
                                          slice_axis=PLOT_AXIS,
                                          slice_idx=PLOT_IDX or 0)
            fig, _ = plot_structured_slice(result,
                                           axis=PLOT_AXIS,
                                           idx=PLOT_IDX,
                                           component=PLOT_COMPONENT,
                                           log_scale=LOG_SCALE,
                                           output_path=png,
                                           show=False,
                                           vmin=VMIN,
                                           vmax=VMAX,
                                           title=f"{stem}  t={t:.4e}")
        else:
            result = read_unstructured(INPUT_FILE, time_idx=ti)
            fig = plot_unstructured(result,
                                    component=PLOT_COMPONENT,
                                    log_scale=LOG_SCALE,
                                    output_path=png,
                                    show=False,
                                    vmin=VMIN,
                                    vmax=VMAX,
                                    title=f"{stem}  t={t:.4e}")

        # Show the frame non-blocking for a moment, then move on
        plt.pause(0.001)
        plt.close(fig)

        print(f"  frame {ti+1:4d}/{len(times)}  t={t:.4e}  -> {os.path.basename(png)}")

    plt.ioff()
    print(f"\nDone.  {len(times)} PNGs in {out_dir}")

    # ── Assemble animation ────────────────────────────────────────────────────
    if ANIMATION_FORMAT is None:
        return

    if ANIMATION_FORMAT == "gif":
        try:
            from PIL import Image
        except ImportError:
            print("ERROR: Pillow not installed.  Run:  pip install Pillow")
            return
        frames = [Image.open(p) for p in png_paths]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.gif")
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=FRAME_DURATION_MS,
            loop=0,
        )
        print(f"GIF saved:  {out_path}")

    elif ANIMATION_FORMAT == "mp4":
        try:
            import imageio
        except ImportError:
            print("ERROR: imageio not installed.  Run:  pip install imageio[ffmpeg]")
            return
        fps = max(1, round(1000 / FRAME_DURATION_MS))
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.mp4")
        with imageio.get_writer(out_path, fps=fps) as writer:
            for p in png_paths:
                writer.append_data(imageio.imread(p))
        print(f"Video saved:  {out_path}  ({fps} fps)")

    else:
        print(f"WARNING: unknown ANIMATION_FORMAT '{ANIMATION_FORMAT}'. Use 'gif', 'mp4', or None.")


# ─────────────────────────────────────────────────────────────────────────────
# RUN  <- Uncomment the operation(s) you want to execute
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 0: print what files exist and their shapes/sizes
    inspect_files()

    # Step 1: plot a single 2D slice with matplotlib
    plot_field()

    # Step 2: plot mid-plane slices along all three axes
    # plot_three_slices()

    # Step 3: save one PNG per time step (structured_time or time_animation)
    # plot_time_series()
