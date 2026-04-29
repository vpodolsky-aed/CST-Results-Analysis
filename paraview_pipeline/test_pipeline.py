"""Quick integration test of the paraview_pipeline package."""
import os
import sys

# File now lives inside paraview_pipeline/ — parent dir holds the package root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from paraview_pipeline.reader import detect_format, get_structured_shape, get_time_steps
from paraview_pipeline.writer import write_vtr, write_vtr_streaming, write_vtu, write_pvd
from paraview_pipeline.reader import read_structured, read_unstructured

BASE = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\Field dump in HDF5"
OUT  = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\vtk_output"
os.makedirs(OUT, exist_ok=True)

files = [f for f in os.listdir(BASE) if f.endswith(".h5")]
print("=== Format detection ===")
for fn in sorted(files):
    path = os.path.join(BASE, fn)
    fmt = detect_format(path)
    times = get_time_steps(path)
    shape = get_structured_shape(path) if fmt == "structured" else "n/a"
    print(f"  {fn}")
    print(f"    format={fmt}  shape={shape}  n_times={None if times is None else len(times)}")

# --- Test 1: structured full read -----------------------------------------
print("\n=== Test 1: structured read + write_vtr ===")
path1 = os.path.join(BASE, "Geometry1FIT67GHzE-field.h5")
r1 = read_structured(path1)
out1 = os.path.join(OUT, "test1_full.vtr")
write_vtr(r1, out1)
print(f"  OK  grid={len(r1.x)}x{len(r1.y)}x{len(r1.z)}  -> {out1}")

# --- Test 2: structured slice (z=7) ---------------------------------------
print("\n=== Test 2: structured z-slice ===")
r2 = read_structured(path1, slice_axis="z", slice_idx=7)
out2 = os.path.join(OUT, "test2_zslice.vtr")
write_vtr(r2, out2)
print(f"  OK  grid={len(r2.x)}x{len(r2.y)}x{len(r2.z)}  -> {out2}")

# --- Test 3: structured subsampled (every 2nd) ----------------------------
print("\n=== Test 3: structured subsample=2 ===")
r3 = read_structured(path1, subsample=2)
out3 = os.path.join(OUT, "test3_sub2.vtr")
write_vtr(r3, out3)
print(f"  OK  grid={len(r3.x)}x{len(r3.y)}x{len(r3.z)}  -> {out3}")

# --- Test 4: streaming writer (same as test1, different code path) --------
print("\n=== Test 4: write_vtr_streaming ===")
out4 = os.path.join(OUT, "test4_streaming.vtr")
write_vtr_streaming(path1, out4)
print(f"  OK  -> {out4}")
s1 = os.path.getsize(out1)
s4 = os.path.getsize(out4)
print(f"  write_vtr size={s1}  streaming size={s4}  match={s1==s4}")

# --- Test 5: unstructured point cloud -------------------------------------
print("\n=== Test 5: unstructured (FEM 67 GHz) ===")
path5 = os.path.join(BASE, "Geometry3FEMOpen67GHzE-field.h5")
r5 = read_unstructured(path5)
out5 = os.path.join(OUT, "test5_fem67.vtu")
write_vtu(r5, out5)
print(f"  OK  N={r5.positions.shape[0]}  -> {out5}")

# --- Test 6: time animation (single step) ---------------------------------
print("\n=== Test 6: time animation step 0 ===")
path6 = os.path.join(BASE, "Geometry1FITTimeAnimationE-field.h5")
times6 = get_time_steps(path6)
r6 = read_unstructured(path6, time_idx=0)
out6 = os.path.join(OUT, "test6_t0.vtu")
write_vtu(r6, out6)
print(f"  OK  N={r6.positions.shape[0]}  t={times6[0]:.3e}  -> {out6}")

# --- Test 7: PVD collection -----------------------------------------------
print("\n=== Test 7: PVD collection ===")
pvd_entries = []
for ti, t in enumerate(times6):
    out_ti = os.path.join(OUT, f"test7_t{ti:03d}.vtu")
    r_ti = read_unstructured(path6, time_idx=ti)
    write_vtu(r_ti, out_ti)
    pvd_entries.append((float(t), f"test7_t{ti:03d}.vtu"))
pvd_path = os.path.join(OUT, "test7_animation.pvd")
write_pvd(pvd_entries, pvd_path)
print(f"  OK  {len(pvd_entries)} steps  -> {pvd_path}")

print("\n=== All tests passed ===")
print(f"Output files in: {OUT}")
