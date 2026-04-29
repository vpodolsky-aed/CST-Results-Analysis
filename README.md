# CST Studio Field Visualization Pipeline

Converts CST Studio 2026 E/H-field HDF5 results to ParaView-compatible VTK files,
bypassing the CST GUI entirely.  Handles files of any size — including >20 GB volumes —
through spatial subsampling and z-slab streaming.

---

## Quick Start

**1. Export HDF5 files from CST** _(skip if you already have `.h5` files)_

Edit the `CST_PROJECTS` list in `export_fields_to_hdf5.py`, then run:
```
python export_fields_to_hdf5.py
```
CST will open briefly, export each field monitor to HDF5, and close.

**2. Check what files you have and their shapes:**
```python
# In main.py, only this line is uncommented:
inspect_files()
```
```
python main.py
```

**3. Open `main.py`, set your file paths and parameters, then uncomment one operation:**
```python
INPUT_FILE = r"C:\...\model_1_FIT_Archive_e-field_(f_67).h5"
SUBSAMPLE  = 4      # 4x subsampling = 64x less RAM than full res
```
Uncomment a call at the bottom:
```python
convert_full_volume()   # or convert_slice(), plot_field(), etc.
```

**4. Open the `.vtr` / `.vtu` / `.pvd` output in ParaView.**

---

## File Structure

```
CST Results Analysis/
  main.py                          <- interactive script (edit & run this)
  export_fields_to_hdf5.py         <- batch export from CST -> HDF5
  README.md                        <- this file
  paraview_pipeline/
    __init__.py
    reader.py                      <- HDF5 reading, format detection
    writer.py                      <- VTK file writing (.vtr, .vtu, .pvd)
    plot.py                        <- matplotlib contour / scatter plots
    convert.py                     <- high-level API + command-line interface
    FUNCTIONS.md                   <- detailed per-function reference
    inspect_h5.py                  <- standalone HDF5 inspector utility
```

---

## Exporting HDF5 Files from CST

`export_fields_to_hdf5.py` automates the export of field monitor results from
CST Studio 2026 without any manual GUI interaction.

**Configuration** (edit at the top of the file):

| Variable | Description |
|---|---|
| `CST_PROJECTS` | List of `.cst` file paths to export. Leave empty `[]` to scan all of `CST_ROOT`. |
| `CST_ROOT` | Root folder scanned when `CST_PROJECTS` is empty. |
| `OUTPUT_DIR` | Where HDF5 files are written. |
| `EXPORT_FREQ_DOMAIN` | Export `.m3d` (frequency-domain) monitors. |
| `EXPORT_TIME_DOMAIN` | Export `.t3D` (time-domain) monitors. |
| `DRY_RUN` | `True` = print discovery only, do not open CST. |

**Usage:**
```
python export_fields_to_hdf5.py
```

- CST opens in a visible window, exports all discovered monitors, then closes.
- Output files are named `<project>_<monitor>.h5` and written to `OUTPUT_DIR`.
- Existing output files are skipped (idempotent).
- All time steps in time-domain monitors are exported automatically.

---

## HDF5 Formats Detected Automatically

| Format | Field shape | Mesh info | Typical source |
|---|---|---|---|
| `structured` | `(nz, ny, nx)` | `Mesh line x/y/z` | FIT/FEM frequency-domain monitor |
| `structured_time` | `(nT, nz, ny, nx)` | `Mesh line x/y/z` + `Times` | FIT time-domain monitor |
| `unstructured` | `(N,)` | `Position` | FEM open-boundary point cloud |
| `time_animation` | `(nT, N)` | `Position` + `Times` | FIT/FEM unstructured time-domain |

---

## `main.py` Operations Reference

| # | Function | Input format | Output | When to use |
|---|---|---|---|---|
| 0 | `inspect_files()` | any | terminal | First step — check shapes and valid indices |
| 1 | `convert_full_volume()` | `structured` | `.vtr` | Full 3D grid; use `SUBSAMPLE≥4` for >4 GB files |
| 2 | `convert_slice()` | `structured` | `.vtr` (2D) | One plane; instant, ~100 KB output |
| 3 | `convert_unstructured_file()` | `unstructured` / `time_animation` | `.vtu` | Point-cloud or single time frame |
| 4 | `convert_all_time_steps()` | `time_animation` | `.vtu` series + `.pvd` | Unstructured time sweep → ParaView animation |
| 5 | `batch_convert_all()` | any (folder) | `.vtr`/`.vtu` + `.pvd` | Convert an entire results folder at once |
| 6 | `plot_field()` | any | matplotlib + `.png` | Quick preview without opening ParaView |
| 7 | `plot_three_slices()` | `structured` | 3× matplotlib + `.png` | 3-D overview via x/y/z mid-planes |
| 8 | `phase_animation()` | `structured` (complex) | `.vtr` series + `.pvd` | Phasor sweep (0–360°) for frequency-domain fields |
| 9 | `convert_structured_time_steps()` | `structured_time` | `.vtr` series + `.pvd` | Full time-domain animation → ParaView |

---

## Key Parameters in `main.py`

| Variable | Default | Description |
|---|---|---|
| `INPUT_DIR` | — | Folder containing `.h5` files |
| `OUTPUT_DIR` | — | Folder where VTK files are written |
| `INPUT_FILE` | — | Specific `.h5` file for single-file operations |
| `SUBSAMPLE` | `1` | Keep every Nth point in each axis. `4` = 64× less RAM. |
| `SLICE_AXIS` | `"z"` | `'x'`, `'y'`, `'z'`, or `None` for full 3D |
| `SLICE_IDX` | `7` | Grid index of the slice (see `inspect_files()` for valid range) |
| `PLOT_COMPONENT` | `"magnitude"` | Quantity to colour-map (see table below) |
| `LOG_SCALE` | `True` | Logarithmic colour axis (recommended for field data) |
| `FORCE_STREAMING` | `False` | Force z-slab streaming even for small files |
| `FORCE_MEMORY` | `False` | Force in-memory path even for large files (safe with ≥64 GB RAM) |
| `WRITE_PVD` | `True` | Write `.pvd` animation index alongside `.vtu`/`.vtr` series |
| `N_PHASE_FRAMES` | `36` | Phase steps for `phase_animation()`. `36` = 10°/frame. |
| `USE_GPU` | `True` | Use CuPy GPU acceleration for `phase_animation()` (falls back to NumPy) |
| `BATCH_PATTERN` | `"*.h5"` | Glob pattern for `batch_convert_all()` |

### `PLOT_COMPONENT` options

| Value | Meaning |
|---|---|
| `"magnitude"` | Total field magnitude \|E\| or \|H\| |
| `"x"`, `"y"`, `"z"` | Component magnitudes \|Ex\|, \|Ey\|, \|Ez\| |
| `"phase_x"`, `"phase_y"`, `"phase_z"` | Phase angle in degrees (frequency-domain only) |

---

## Memory Guide for Large Files

A 20 GB HDF5 file has roughly 833 M grid points (6 float32 per point × 24 B each).

| `SUBSAMPLE` | Points kept | Output `.vtr` size | RAM needed |
|---|---|---|---|
| 1 | 100% | ~13 GB | ~26 GB (two float32 arrays) |
| 2 | 12.5% | ~1.7 GB | ~3.3 GB |
| 4 | 1.6% | ~200 MB | ~400 MB |
| 8 | 0.2% | ~25 MB | ~50 MB |

With `SUBSAMPLE=1` and no slice, the pipeline automatically enables z-slab streaming
so **only one z-plane is in RAM at a time** (~24 MB for a 1000×1000 grid).
Disk I/O is the bottleneck in that case, not RAM.

`FORCE_MEMORY=True` bypasses auto-streaming and loads the full field in one shot.
Safe when you have ≥64 GB RAM and want the fastest possible conversion.

---

## Opening Results in ParaView

| File type | How to open |
|---|---|
| Single `.vtr` / `.vtu` | File → Open |
| `.pvd` animation | File → Open the `.pvd` file; use the Play button to animate |
| Multiple `.vtr` files | File → Open → select all → ParaView groups them automatically |

**Recommended filters after loading a `.vtr`:**

1. Apply **Threshold** on `E_magnitude` to mask near-zero regions
2. Apply **Glyph** on `E_vector` to show field direction arrows
3. Apply **Slice** for an interactive cross-section

---

## Command-Line Interface

The same operations are available from the terminal:

```bash
# Single structured file
python -m paraview_pipeline.convert structured INPUT.h5 OUTPUT.vtr --subsample 4

# 2D slice
python -m paraview_pipeline.convert structured INPUT.h5 slice.vtr --slice-axis z --slice-idx 7

# Force streaming (large files)
python -m paraview_pipeline.convert structured BigFile.h5 out.vtr --stream

# Force in-memory (with plenty of RAM)
python -m paraview_pipeline.convert structured BigFile.h5 out.vtr --force-memory

# Unstructured / point cloud
python -m paraview_pipeline.convert unstructured INPUT.h5 OUTPUT.vtu

# Unstructured time-animation -> .vtu series
python -m paraview_pipeline.convert time-series INPUT.h5 output_folder/

# Structured time-domain -> .vtr series + .pvd
python -m paraview_pipeline.convert structured-time-series INPUT.h5 output_folder/ --subsample 2

# Phase animation (frequency-domain phasor sweep)
python -m paraview_pipeline.convert phase-animation INPUT.h5 output_folder/ --frames 36

# Entire folder (all formats)
python -m paraview_pipeline.convert batch INPUT_DIR/ OUTPUT_DIR/ --subsample 4 --pattern "*E-field.h5"

# matplotlib plot (no ParaView needed)
python -m paraview_pipeline.convert plot INPUT.h5 --slice-axis z --slice-idx 7 --log --output fig.png
```

---

## Installation Requirements

```bash
pip install h5py numpy matplotlib
```

ParaView is a separate download: [https://www.paraview.org/download/](https://www.paraview.org/download/)

CST Studio Suite 2026 must be installed to export HDF5 files via `export_fields_to_hdf5.py`.
The pipeline reads HDF5 files directly — CST does not need to be running during conversion.
