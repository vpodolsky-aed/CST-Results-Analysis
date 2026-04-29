# Function Reference — `paraview_pipeline`

Each module is described below with its functions, parameters, return values, and
worked examples.  For a narrative overview see `../README.md`.

---

## `reader.py` — Reading CST HDF5 files

### `detect_format(filepath)`

Reads only the shape of the E/H-Field dataset and returns a string.

| Return value | Condition |
|---|---|
| `"structured"` | Field shape is `(nz, ny, nx)` — rectilinear grid, frequency-domain |
| `"structured_time"` | Field shape is `(nT, nz, ny, nx)` — rectilinear grid, time-domain |
| `"unstructured"` | Field shape is `(N,)` — arbitrary point positions |
| `"time_animation"` | Field shape is `(nT, N)` — time-domain frames on point cloud |

```python
fmt = detect_format("67GHz-E-field.h5")       # -> "structured"
fmt = detect_format("time_monitor-E-field.h5") # -> "structured_time"
```

---

### `get_structured_shape(filepath)`

Returns `(nz, ny, nx)` without reading any field data.
Use this to find valid `slice_idx` ranges before converting.

Works for both `structured` (`nz, ny, nx`) and `structured_time` (`nT, nz, ny, nx`)
files — always returns the spatial dimensions only.

```python
nz, ny, nx = get_structured_shape("67GHz-E-field.h5")
# e.g.  nz=14, ny=24, nx=32
# valid SLICE_IDX for SLICE_AXIS='z' is 0..13

nz, ny, nx = get_structured_shape("time_monitor-E-field.h5")
# returns spatial dims (nz, ny, nx) regardless of number of time steps
```

---

### `get_time_steps(filepath)`

Returns a `(nT,)` float64 array of time values (in seconds) for time-domain files,
or `None` for frequency-domain snapshots.

Works for both `time_animation` and `structured_time` formats.

```python
times = get_time_steps("TimeAnimation.h5")       # unstructured time-animation
times = get_time_steps("time_monitor-E-field.h5") # structured_time
# e.g.  array([0., 2.5e-12, 5e-12, ...])
# len(times) tells you how many frames to loop over
```

---

### `read_structured(filepath, subsample=1, slice_axis=None, slice_idx=0)`

Loads a rectilinear-grid HDF5 file and returns a `StructuredResult`.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | str | Path to the `.h5` file |
| `subsample` | int | Keep every Nth point in all axes.  `1` = full resolution. |
| `slice_axis` | `'x'`, `'y'`, `'z'`, or `None` | Axis perpendicular to the extracted plane. `None` = full 3D. |
| `slice_idx` | int | Grid index along `slice_axis`. |

**Returns: `StructuredResult`**

| Field | Type | Description |
|---|---|---|
| `.x`, `.y`, `.z` | `(nx,)`, `(ny,)`, `(nz,)` float32 | Mesh coordinates in metres |
| `.Ex`, `.Ey`, `.Ez` | `(nz, ny, nx)` complex64 or float32 | Field components |
| `.field_type` | `"E"` or `"H"` | Which field |
| `.is_complex` | bool | `True` for frequency-domain, `False` for time-domain |

HDF5 reads only the selected region — if you request a 2D slice or a subsampled
grid, only those bytes are loaded from disk.

```python
# Full 3D (fine for files that fit in RAM)
r = read_structured("67GHz.h5")

# Every 4th point — 64x less RAM
r = read_structured("BigFile.h5", subsample=4)

# Single z-plane at index 7
r = read_structured("67GHz.h5", slice_axis="z", slice_idx=7)

# Access field values
import numpy as np
E_mag = np.sqrt(np.abs(r.Ex)**2 + np.abs(r.Ey)**2 + np.abs(r.Ez)**2)
print(f"Max |E| = {E_mag.max():.2f} V/m")
```

> **Large file advice:** for files larger than ~4 GB prefer
> `write_vtr_streaming()` which reads one z-plane at a time and never holds
> more than one slab in RAM simultaneously.

---

### `read_structured_time(filepath, time_idx=None, subsample=1, slice_axis=None, slice_idx=0)`

Loads one time step from a `structured_time` (`nT, nz, ny, nx`) HDF5 file and
returns a `StructuredResult`.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | str | Path to the `.h5` file |
| `time_idx` | int or `None` | Which time step to load. `None` defaults to 0. |
| `subsample` | int | Keep every Nth point in all axes. |
| `slice_axis` | `'x'`, `'y'`, `'z'`, or `None` | Axis for 2D slice; `None` = full 3D. |
| `slice_idx` | int | Grid index along `slice_axis`. |

Returns a `StructuredResult` with `is_complex=False` (time-domain data is always real).
Use `get_time_steps()` to retrieve the full `Times` array before looping.

```python
from paraview_pipeline.reader import read_structured_time, get_time_steps

times = get_time_steps("time_monitor-E-field.h5")
print(f"{len(times)} time steps, first={times[0]:.3e} s, last={times[-1]:.3e} s")

# Load step 10
r = read_structured_time("time_monitor-E-field.h5", time_idx=10)
print(r.Ex.shape)   # (nz, ny, nx)

# Load step 0 with 2x subsampling
r = read_structured_time("time_monitor-E-field.h5", time_idx=0, subsample=2)
```

> **Large file advice:** for many frames or large grids use
> `convert_structured_time_series()` which calls `write_vtr_streaming()` with
> `time_idx` and never holds more than one z-slab in RAM.

---

### `read_unstructured(filepath, time_idx=None)`

Loads an unstructured or time-animation HDF5 file.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | str | Path to the `.h5` file |
| `time_idx` | int or `None` | Frame index for time-animation files. `None` defaults to 0. |

**Returns: `UnstructuredResult`**

| Field | Type | Description |
|---|---|---|
| `.positions` | `(N, 3)` float32 | x, y, z coordinates in metres |
| `.Fx`, `.Fy`, `.Fz` | `(N,)` complex64 or float32 | Field components |
| `.field_type` | `"E"` or `"H"` | Which field |
| `.is_complex` | bool | |

```python
# Frequency-domain point cloud
r = read_unstructured("FEM67GHz-E-field.h5")

# Time-domain, frame 10
r = read_unstructured("TimeAnim-E-field.h5", time_idx=10)
```

---

---

## `writer.py` — Writing VTK files

All writers use **binary-appended VTK XML format** with `UInt64` length headers.
This is ~6× more compact than ASCII and can be read in chunks — there is no size
limit imposed by the writer itself.

Each `.vtr` / `.vtu` contains two PointData arrays:
- `E_magnitude` (or `H_magnitude`) — scalar `|F|` in V/m or A/m
- `E_vector` (or `H_vector`) — 3-component real part of the field, for glyph plots

---

### `write_vtr(result, output_path)`

Write a `StructuredResult` to a `.vtr` RectilinearGrid file.
Computes all arrays in memory at once — only suitable when the data already
fits in RAM (i.e. after reading with `read_structured()` or `read_structured_time()`).

```python
from paraview_pipeline.reader import read_structured
from paraview_pipeline.writer import write_vtr

r = read_structured("67GHz.h5", subsample=2)
write_vtr(r, "output/67GHz_sub2.vtr")
```

---

### `write_vtr_streaming(filepath, output_path, subsample=1, slice_axis=None, slice_idx=0, time_idx=None)`

Memory-safe conversion directly from HDF5 to `.vtr` without loading the whole
field into RAM.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | str | Path to the `.h5` file |
| `output_path` | str | Destination `.vtr` file |
| `subsample` | int | Keep every Nth point in all axes |
| `slice_axis` | `'x'`, `'y'`, `'z'`, or `None` | Axis for 2D slice |
| `slice_idx` | int | Grid index along `slice_axis` |
| `time_idx` | int or `None` | For `structured_time` (4D) files — which time step to stream. Ignored for plain `structured` (3D) files. |

**Workflow inside this function:**

1. Opens the HDF5 file and reads only the axis coordinates (tiny).
2. Calculates the exact output dimensions and VTK byte offsets.
3. Writes the XML header.
4. Loops over z-planes one at a time:
   - Reads one slab from HDF5 (`~24 MB` for a 1000×1000 grid).
   - Computes `|F|` and real components for that slab.
   - Writes the magnitude bytes directly to the output file.
   - Buffers the vector bytes to a temporary file.
5. Appends the buffered vector data, then coordinate arrays.

At no point is more than one z-slab plus one vector-slab-buffer in RAM.

```python
from paraview_pipeline.writer import write_vtr_streaming

# Full 3D, streaming — safe for any file size
write_vtr_streaming("BigFile.h5", "out.vtr")

# Subsampled (HDF5 only reads selected points — equally memory-safe)
write_vtr_streaming("BigFile.h5", "out_sub4.vtr", subsample=4)

# z-slice (instant)
write_vtr_streaming("BigFile.h5", "slice_z50.vtr", slice_axis="z", slice_idx=50)

# One time step from a structured_time (4D) file
write_vtr_streaming("time_monitor.h5", "t0010.vtr", time_idx=10)
```

---

### `write_vtu(result, output_path)`

Write an `UnstructuredResult` to a `.vtu` UnstructuredGrid file.
Each point becomes a `VTK_VERTEX` cell.

```python
from paraview_pipeline.reader import read_unstructured
from paraview_pipeline.writer import write_vtu

r = read_unstructured("FEM67GHz.h5")
write_vtu(r, "output/FEM67GHz.vtu")
```

In ParaView, apply **Point Gaussian** or **Glyph** filters to visualise the
point-cloud data.

---

### `write_pvd(entries, output_path)`

Write a ParaView Data Collection (`.pvd`) file that groups multiple `.vtr` or
`.vtu` files into an animation timeline.

| Parameter | Type | Description |
|---|---|---|
| `entries` | list of `(timestep, relative_path)` tuples | Each tuple is one frame |
| `output_path` | str | Destination `.pvd` file path |

The `timestep` value sets the position on ParaView's time slider. For
frequency sweeps, use frequency in Hz. For time-domain, use time in seconds.
For phase animations, use phase angle in radians.

```python
from paraview_pipeline.writer import write_pvd

entries = [
    (67e9,  "step_67GHz.vtr"),
    (70e9,  "step_70GHz.vtr"),
    (73e9,  "step_73GHz.vtr"),
]
write_pvd(entries, "output/sweep.pvd")
# Open sweep.pvd in ParaView — use the Play button to animate
```

---

---

## `plot.py` — Matplotlib field plots

No ParaView needed. Useful for a quick sanity check or paper figures.

---

### `plot_structured_slice(result, axis="z", idx=None, component="magnitude", log_scale=False, output_path=None, show=True)`

Plot a 2-D colour-map of a cross-section of a `StructuredResult`.

| Parameter | Type | Description |
|---|---|---|
| `result` | `StructuredResult` | From `read_structured()` or `read_structured_time()` |
| `axis` | `"x"`, `"y"`, or `"z"` | Axis perpendicular to the plot plane |
| `idx` | int or `None` | Grid index along `axis`. `None` = midpoint. |
| `component` | str | See table below |
| `log_scale` | bool | Logarithmic colour axis |
| `output_path` | str or `None` | Save PNG/PDF here if given |
| `show` | bool | Call `plt.show()` (set `False` for headless/batch use) |

**Returns** `(fig, ax)` — standard matplotlib objects for further customisation.

**`component` options:**

| Value | Shown |
|---|---|
| `"magnitude"` | \|F\| — total field magnitude + unit-direction quiver overlay |
| `"x"` / `"y"` / `"z"` | \|Fx\|, \|Fy\|, \|Fz\| — individual components |
| `"phase_x"` / `"phase_y"` / `"phase_z"` | Phase angle in degrees (frequency-domain only) |

```python
from paraview_pipeline.reader import read_structured
from paraview_pipeline.plot   import plot_structured_slice

r = read_structured("67GHz.h5")

# z-slice at index 7, log scale, save PNG
fig, ax = plot_structured_slice(r, axis="z", idx=7,
                                 component="magnitude",
                                 log_scale=True,
                                 output_path="z7_mag.png")

# x-component only, linear scale, midplane
fig, ax = plot_structured_slice(r, axis="y", component="x", log_scale=False)

# Customise after the fact
ax.set_title("My custom title")
fig.savefig("custom.png", dpi=300)
```

---

### `plot_unstructured(result, component="magnitude", log_scale=False, output_path=None, show=True)`

Scatter plot of point-cloud data — x-y projection and x-z projection side by side.

| Parameter | Type | Description |
|---|---|---|
| `result` | `UnstructuredResult` | From `read_unstructured()` |
| `component` | str | Same options as above |
| `log_scale` | bool | Logarithmic colour axis |
| `output_path` | str or `None` | Save PNG here |
| `show` | bool | |

**Returns** `fig`.

```python
from paraview_pipeline.reader import read_unstructured
from paraview_pipeline.plot   import plot_unstructured

r = read_unstructured("FEM67GHz.h5")
fig = plot_unstructured(r, component="magnitude", log_scale=True,
                        output_path="fem_scatter.png")
```

---

---

## `convert.py` — High-level API and CLI

These functions are thin wrappers that wire together reader + writer with
sensible defaults and progress printing.

---

### `convert_structured(input_h5, output_vtr, subsample=1, slice_axis=None, slice_idx=0, streaming=False, force_memory=False)`

Convert one structured HDF5 file to `.vtr`.

Automatically switches to `write_vtr_streaming()` when:
- `streaming=True`, **or**
- `slice_axis=None` and `subsample=1` and estimated output > 1 GB.

`force_memory=True` bypasses auto-streaming and always loads the full field
into RAM. Safe when you have ≥64 GB RAM and want the fastest possible conversion.

```python
from paraview_pipeline.convert import convert_structured

convert_structured("67GHz.h5", "out/67GHz.vtr")
convert_structured("BigFile.h5", "out/sub4.vtr", subsample=4)
convert_structured("BigFile.h5", "out/slice.vtr", slice_axis="z", slice_idx=50)
convert_structured("BigFile.h5", "out/full.vtr", streaming=True)
convert_structured("BigFile.h5", "out/full.vtr", force_memory=True)  # ≥64 GB RAM
```

---

### `convert_unstructured(input_h5, output_vtu, time_idx=None)`

Convert one unstructured or time-animation HDF5 snapshot to `.vtu`.

```python
from paraview_pipeline.convert import convert_unstructured

convert_unstructured("FEM67GHz.h5", "out/FEM67GHz.vtu")
convert_unstructured("TimeAnim.h5", "out/t005.vtu", time_idx=5)
```

---

### `convert_time_series(input_h5, output_dir, write_pvd_file=True)`

Convert all time frames in a `time_animation` HDF5 to individual `.vtu` files.
Writes a `.pvd` collection so all frames load as one animation in ParaView.

```python
from paraview_pipeline.convert import convert_time_series

convert_time_series("TimeAnim.h5", "out/anim/")
# Creates: out/anim/TimeAnim_t0000.vtu, ..._t0133.vtu, TimeAnim.pvd
```

**In ParaView:** File → Open → select `TimeAnim.pvd` → click Play.

---

### `convert_structured_time_series(input_h5, output_dir, subsample=1, write_pvd_file=True, streaming=False, force_memory=False)`

Convert every time step in a `structured_time` (`nT, nz, ny, nx`) HDF5 to
individual `.vtr` files. Writes a `.pvd` collection for ParaView animation.

This is the structured-grid equivalent of `convert_time_series()`. Use it for
time-domain field monitors exported from CST as 4D arrays.

| Parameter | Type | Description |
|---|---|---|
| `input_h5` | str | Path to a `structured_time` HDF5 file |
| `output_dir` | str | Directory where `.vtr` frames and `.pvd` are written |
| `subsample` | int | Keep every Nth point in all axes (applied per frame) |
| `write_pvd_file` | bool | Write a `.pvd` collection (default `True`) |
| `streaming` | bool | Force z-slab streaming per frame (low RAM) |
| `force_memory` | bool | Force in-memory path per frame (fast with ≥64 GB RAM) |

Auto-streaming is enabled when `subsample=1` and the estimated per-frame
output exceeds 1 GB.

Returns list of output `.vtr` paths.

```python
from paraview_pipeline.convert import convert_structured_time_series

convert_structured_time_series(
    "time_monitor-E-field.h5",
    "out/time_monitor/",
    subsample=2,          # 8x less RAM/disk per frame
    write_pvd_file=True,
)
# Creates: out/time_monitor/time_monitor-E-field_t0000.vtr ... _t0133.vtr
#          out/time_monitor/time_monitor-E-field.pvd
```

**In ParaView:** File → Open → select the `.pvd` file → click Play.

---

### `convert_phase_animation(input_h5, output_dir, n_frames=36, subsample=1, write_pvd_file=True, use_gpu=True)`

Generate a phasor-sweep animation from a frequency-domain structured file.

Computes `E(θ) = Re(E_complex × exp(j×θ))` for θ in [0°, 360°) and writes
`n_frames` `.vtr` files plus a `.pvd` collection.

Raises `ValueError` if the file is not `structured` format or contains real
(time-domain) data.

| Parameter | Type | Description |
|---|---|---|
| `input_h5` | str | Path to a `structured` frequency-domain HDF5 file |
| `output_dir` | str | Directory where `.vtr` frames and `.pvd` are written |
| `n_frames` | int | Number of phase steps (default 36 = 10° per frame) |
| `subsample` | int | Keep every Nth point in all axes |
| `write_pvd_file` | bool | Write a `.pvd` collection (default `True`) |
| `use_gpu` | bool | Attempt CuPy GPU acceleration; falls back to NumPy silently |

Returns the `.pvd` path if `write_pvd_file=True`, else `None`.

```python
from paraview_pipeline.convert import convert_phase_animation

convert_phase_animation(
    "67GHz-E-field.h5",
    "out/phase_anim/",
    n_frames=36,    # 10 deg/frame — good default
    subsample=2,    # halve resolution if file is large
    use_gpu=True,   # uses CuPy if available, falls back to NumPy
)
# Creates: out/phase_anim/67GHz-E-field_phase000deg.vtr ... _phase350deg.vtr
#          out/phase_anim/67GHz-E-field_phase_animation.pvd
```

**In ParaView:** File → Open → select the `_phase_animation.pvd` file → click Play.

---

### `convert_batch(input_dir, output_dir, pattern="*.h5", subsample=1, write_pvd_file=True, streaming=False)`

Batch-convert all matching HDF5 files in `input_dir`.

- `structured` files → `.vtr`, grouped by field type into a per-type `.pvd`
- `structured_time` files → per-file subfolder + `.pvd` via `convert_structured_time_series()`
- `time_animation` files → per-file subfolder + `.pvd` via `convert_time_series()`
- `unstructured` files → `.vtu`, grouped by field type into a per-type `.pvd`

```python
from paraview_pipeline.convert import convert_batch

convert_batch(
    input_dir   = r"C:\Sim\Results",
    output_dir  = r"C:\Sim\VTK",
    pattern     = "*E-field.h5",  # only E-field results
    subsample   = 4,              # 64x less RAM per file
    write_pvd_file = True,
)
```

---

### Command-line usage

```
python -m paraview_pipeline.convert <subcommand> [options]
```

| Subcommand | Description |
|---|---|
| `structured INPUT.h5 OUTPUT.vtr [--subsample N] [--slice-axis z] [--slice-idx 0] [--stream] [--force-memory]` | Single structured file |
| `unstructured INPUT.h5 OUTPUT.vtu [--time-idx T]` | Single unstructured snapshot |
| `time-series INPUT.h5 OUTPUT_DIR/ [--no-pvd]` | All frames from unstructured time-animation |
| `structured-time-series INPUT.h5 OUTPUT_DIR/ [--subsample N] [--no-pvd] [--stream] [--force-memory]` | All frames from structured time-domain monitor |
| `phase-animation INPUT.h5 OUTPUT_DIR/ [--frames N] [--subsample N] [--no-pvd] [--no-gpu]` | Phasor sweep animation from frequency-domain file |
| `batch INPUT_DIR/ OUTPUT_DIR/ [--pattern *.h5] [--subsample N] [--no-pvd] [--stream]` | Whole folder (all formats) |
| `plot INPUT.h5 [--slice-axis z] [--slice-idx 0] [--component magnitude] [--log] [--output fig.png] [--no-show]` | matplotlib plot |
