"""Headless plot tests — verifies matplotlib output for structured and unstructured data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")  # headless, no display required

from paraview_pipeline.reader import read_structured, read_unstructured
from paraview_pipeline.plot import plot_structured_slice, plot_unstructured

BASE = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\Field dump in HDF5"
OUT  = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\vtk_output"
os.makedirs(OUT, exist_ok=True)

r = read_structured(os.path.join(BASE, "Geometry1FIT67GHzE-field.h5"))

fig, ax = plot_structured_slice(r, axis="z", idx=7, component="magnitude",
    log_scale=True, output_path=os.path.join(OUT, "plot_z7_log.png"), show=False)
print("z-slice magnitude (log) OK")

fig, ax = plot_structured_slice(r, axis="y", component="magnitude",
    output_path=os.path.join(OUT, "plot_ymid_mag.png"), show=False)
print("y-slice magnitude OK")

fig, ax = plot_structured_slice(r, axis="z", idx=7, component="phase_x",
    output_path=os.path.join(OUT, "plot_z7_phase.png"), show=False)
print("z-slice phase_x OK")

r2 = read_unstructured(os.path.join(BASE, "Geometry3FEMOpen67GHzE-field.h5"))
fig3 = plot_unstructured(r2, component="magnitude", log_scale=True,
    output_path=os.path.join(OUT, "plot_fem_scatter.png"), show=False)
print("unstructured scatter OK")

print("\nAll plot tests passed")
