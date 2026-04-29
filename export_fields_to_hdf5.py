"""
export_fields_to_hdf5.py
========================
Batch-exports CST field monitor results (.m3d, .t3D) to HDF5 without
requiring any manual GUI interaction.

How it works
------------
1. Scans CST_PROJECTS for .cst files.
2. For each project, looks in <project_stem>/Result/ for completed
   field monitor files (.m3d = frequency domain, .t3D = time domain).
3. Launches CST Studio Suite once in quiet mode (may briefly show
   a minimised window on Windows — no clicks required from you).
4. Opens each project, calls the official VBA ASCIIExport API to write
   every discovered field monitor directly to HDF5, then closes it.
5. Output files land in OUTPUT_DIR, ready for the paraview_pipeline.

Resulting HDF5 files are drop-in replacements for files in INPUT_DIR
in main.py — no format changes needed.

Usage
-----
  python export_fields_to_hdf5.py

Edit the CONFIGURATION section below before running.
"""

import os
import re
import time
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION  <- Edit these
# ---------------------------------------------------------------------------

# Root folder that contains your .cst project files.
# Used for full-batch mode (when CST_PROJECTS is empty).
# The script walks this tree and exports every project that has completed
# field monitor results.
CST_ROOT = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files"

# Single-project (or subset) mode.
# List specific .cst file paths here to export only those projects.
# Leave empty ( [] ) to fall back to scanning the full CST_ROOT tree.
#
# Examples:
#   CST_PROJECTS = [r"C:\...\model_1_FIT_Archive.cst"]           # one project
#   CST_PROJECTS = [r"C:\...\model_1_FIT_Archive.cst",
#                   r"C:\...\Cloud2\Scooter_Curved_Anode.cst"]   # two projects
#   CST_PROJECTS = []                                             # full batch
CST_PROJECTS = [
    r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\model_1_FIT_Archive.cst",
]

# Where to write the exported HDF5 files.
OUTPUT_DIR = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\Field dump in HDF5"


# Which monitor types to export.  Both are enabled by default.
EXPORT_FREQ_DOMAIN  = True   # .m3d files  (e-field (f=67), etc.)
EXPORT_TIME_DOMAIN  = True   # .t3D files  (e-field (t=0..20), etc.)

# Dry run: if True, prints what would be exported but does NOT open CST.
# Set to False when you are ready to actually run.
DRY_RUN = False

# ---------------------------------------------------------------------------
# CST path setup  (do not edit unless your CST is installed elsewhere)
# ---------------------------------------------------------------------------

_CST_AMD64  = r"C:\Program Files\CST Studio Suite 2026\AMD64"
_CST_PYLIB  = r"C:\Program Files\CST Studio Suite 2026\AMD64\python_cst_libraries"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _find_result_dir(cst_file: Path) -> Path | None:
    """
    Return the Result/ directory for a .cst project file, or None if not found.

    CST project layout (2025+):
        workspace/
            ProjectName.cst        <- project file (this is the argument)
            ProjectName/           <- data directory
                Result/            <- field monitor files live here
    """
    stem = cst_file.stem
    result_dir = cst_file.parent / stem / "Result"
    return result_dir if result_dir.is_dir() else None


def _stem_from_m3d(filename: str) -> str:
    """
    Strip the port/mode suffix from an .m3d filename stem (used for output name).
    e-field (f=67)_1,1.m3d  ->  e-field (f=67)
    """
    return re.sub(r"_\d+,\d+$", "", Path(filename).stem)


def _tree_name_from_m3d(filename: str) -> str:
    """
    Build the CST navigation tree display name from an .m3d filename.
    The tree appends ' [port]' from the _port,mode suffix in the filename.
    e-field (f=67)_1,1.m3d  ->  e-field (f=67) [1]
    """
    stem = Path(filename).stem
    m = re.search(r"_(\d+),\d+$", stem)
    base = re.sub(r"_\d+,\d+$", "", stem)
    return f"{base} [{m.group(1)}]" if m else base


def _stem_from_t3d(filename: str) -> str:
    """
    Strip the run-id suffix from a .t3D filename stem (used for output name).
    e-field (t=0.5..2(0.3))_1.t3D  ->  e-field (t=0.5..2(0.3))
    """
    return re.sub(r"_\d+$", "", Path(filename).stem)


def _tree_name_from_t3d(filename: str) -> str:
    """
    Build the CST navigation tree display name from a .t3D filename.
    e-field (t=0..20(0.15))_1.t3D  ->  e-field (t=0..20(0.15)) [1]
    """
    stem = Path(filename).stem
    m = re.search(r"_(\d+)$", stem)
    base = re.sub(r"_\d+$", "", stem)
    return f"{base} [{m.group(1)}]" if m else base


def _tree_path(tree_name: str) -> str:
    """
    Build the full CST navigation tree path from a monitor tree display name.
    e-field (f=67) [1]  ->  2D/3D Results\\E-Field\\e-field (f=67) [1]
    h-field (f=67) [1]  ->  2D/3D Results\\H-Field\\h-field (f=67) [1]
    """
    if tree_name.lower().startswith("h-field"):
        folder = "H-Field"
    else:
        folder = "E-Field"
    return rf"2D/3D Results\{folder}\{tree_name}"


def _t3d_step_count(t3d_file: Path) -> int | None:
    """
    Parse the companion .rex XML file to get the number of time steps.

    The rex file has a <TimeSampling start="..." end="..." step="..."/> element.
    Returns None if the rex file is missing or the element is absent.
    """
    stem = t3d_file.stem                          # e.g. "e-field (t=0..20(0.15))_1"
    rex = t3d_file.with_name(stem + "_t3D.rex")  # e.g. "…_1_t3D.rex"
    if not rex.exists():
        return None
    try:
        root = ET.parse(rex).getroot()
        ts = root.find("TimeSampling")
        if ts is None:
            return None
        start = float(ts.get("start", 0))
        end   = float(ts.get("end", 0))
        step  = float(ts.get("step", 0))
        if step == 0:
            return None
        return round((end - start) / step) + 1
    except Exception:
        return None


def _safe_filename(s: str) -> str:
    """Make a string safe for use in a filename."""
    s = re.sub(r"[;=\s]+", "_", s)
    s = re.sub(r"[^\w\-\.()\[\]]", "", s)
    return s.strip("_")


def _output_path(cst_file: Path, monitor_stem: str) -> Path:
    """
    Build the output HDF5 path using the project's path relative to CST_ROOT
    so that two projects with the same stem (e.g. Cloud2/Foo.cst vs Foo.cst)
    produce distinct output filenames.

    Cloud2/Scooter_Curved_Anode.cst + e-field (f=67)
      ->  <OUTPUT_DIR>/Cloud2_Scooter_Curved_Anode_e-field_(f_67).h5
    """
    try:
        rel = cst_file.relative_to(Path(CST_ROOT))
    except ValueError:
        rel = cst_file

    # Build a flat prefix from the relative path parts (excluding .cst extension).
    # Deduplicate consecutive identical parts that arise from CST's nested
    # workspace structure (e.g. Sim1_run2/Sim1_run2.cst -> just Sim1_run2).
    parts = list(rel.parts)
    parts[-1] = rel.stem  # drop .cst
    deduped = [parts[0]] + [p for i, p in enumerate(parts[1:], 1) if p != parts[i - 1]]
    prefix = "_".join(p for p in deduped if p)

    safe_monitor = _safe_filename(monitor_stem)
    name = f"{prefix}_{safe_monitor}.h5"
    return Path(OUTPUT_DIR) / name


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_projects(root: str):
    """
    Walk root and yield (cst_file, result_dir, monitors) tuples.

    monitors is a list of dicts:
        { "stem": str, "tree_path": str, "file": Path, "type": "m3d"|"t3d" }
    """
    root_path = Path(root)
    for cst_file in sorted(root_path.rglob("*.cst")):
        if not cst_file.is_file():
            continue  # skip if it's a directory masquerading as .cst

        result_dir = _find_result_dir(cst_file)
        if result_dir is None:
            continue

        monitors = []

        if EXPORT_FREQ_DOMAIN:
            for f in sorted(result_dir.glob("*.m3d")):
                stem = _stem_from_m3d(f.name)
                monitors.append({
                    "stem":      stem,
                    "tree_path": _tree_path(_tree_name_from_m3d(f.name)),
                    "file":      f,
                    "type":      "m3d",
                    "out":       _output_path(cst_file, stem),
                })

        if EXPORT_TIME_DOMAIN:
            for f in sorted(result_dir.glob("*.t3D")):
                stem = _stem_from_t3d(f.name)
                monitors.append({
                    "stem":      stem,
                    "tree_path": _tree_path(_tree_name_from_t3d(f.name)),
                    "file":      f,
                    "type":      "t3d",
                    "n_steps":   _t3d_step_count(f),
                    "out":       _output_path(cst_file, stem),
                })

        if monitors:
            yield cst_file, result_dir, monitors


# ---------------------------------------------------------------------------
# Export (requires CST)
# ---------------------------------------------------------------------------

def _export_one(model3d, monitor: dict) -> bool:
    """
    Export a single monitor via direct Python IPC calls.

    With CST running in normal (non-quiet) mode the GUI is live, so
    SelectTreeItem properly switches the active view and ASCIIExport.Execute
    finds the field result.
    """
    try:
        selected = model3d.SelectTreeItem(monitor["tree_path"])
        log.info("    SelectTreeItem(%s) -> %s", monitor["tree_path"], selected)
        try:
            model3d.DoEvents()
        except Exception as e:
            log.debug("    DoEvents: %s", e)
        time.sleep(2)
        with model3d.ASCIIExport as exp:
            exp.Reset()
            exp.FileName(str(monitor["out"]))
            exp.SetFileType("hdf5")
            if monitor["type"] == "t3d":
                n = monitor.get("n_steps")
                if n:
                    log.info("    Setting sample range 0..%d", n - 1)
                    exp.SetSampleRange(0, n - 1)
            exp.Execute()
        return True
    except Exception as exc:
        log.error("    Export failed: %s", exc)
        return False


def run_export(projects):
    """
    Launch CST in normal (visible) mode and export all discovered monitors.

    Quiet mode is intentionally left OFF: in quiet/headless mode SelectTreeItem
    does not switch the active display view, so ASCIIExport.Execute fails with
    'not available for the current view (0) Components'.  With a live GUI window
    the view switch works correctly.  CST will briefly appear in the taskbar but
    requires no interaction.
    """
    import sys
    sys.path.insert(0, _CST_PYLIB)
    sys.path.insert(0, _CST_AMD64)

    try:
        from cst.interface import DesignEnvironment
    except ImportError as e:
        log.error("Cannot import cst.interface: %s", e)
        log.error("Make sure CST Studio Suite 2026 is installed at %s", _CST_AMD64)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Launching CST Studio Suite (normal mode -- window will appear) ...")
    de = DesignEnvironment.new()
    # Quiet mode deliberately disabled so SelectTreeItem activates the result view.
    log.info("CST started (PID %d)", de.pid())

    try:
        for cst_file, _, monitors in projects:
            log.info("")
            log.info("Project: %s", cst_file)
            log.info("  Opening ...")

            try:
                prj = de.open_project(str(cst_file))
            except Exception as exc:
                log.error("  Could not open project: %s", exc)
                continue

            m = prj.model3d
            ok = fail = 0

            for mon in monitors:
                label = f"  [{mon['type'].upper()}] {mon['stem']}"
                if mon["out"].exists():
                    log.info("%s  (skipping -- output already exists)", label)
                    ok += 1
                    continue

                log.info("%s  -> %s", label, mon["out"].name)
                if _export_one(m, mon):
                    ok += 1
                else:
                    fail += 1

            log.info("  Done: %d exported, %d failed", ok, fail)
            prj.close()

    finally:
        log.info("\nClosing CST ...")
        de.close()
        log.info("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_projects():
    """
    Return the list of (cst_file, result_dir, monitors) tuples to process.

    If CST_PROJECTS is non-empty, use exactly those .cst files (in order).
    Otherwise, scan the full CST_ROOT tree.
    """
    if CST_PROJECTS:
        results = []
        for path_str in CST_PROJECTS:
            cst_file = Path(path_str)
            if not cst_file.is_file():
                log.warning("CST_PROJECTS entry not found, skipping: %s", cst_file)
                continue
            result_dir = _find_result_dir(cst_file)
            if result_dir is None:
                log.warning("No Result/ folder for %s, skipping", cst_file.name)
                continue
            monitors = []
            if EXPORT_FREQ_DOMAIN:
                for f in sorted(result_dir.glob("*.m3d")):
                    stem = _stem_from_m3d(f.name)
                    monitors.append({
                        "stem":      stem,
                        "tree_path": _tree_path(_tree_name_from_m3d(f.name)),
                        "file":      f,
                        "type":      "m3d",
                        "out":       _output_path(cst_file, stem),
                    })
            if EXPORT_TIME_DOMAIN:
                for f in sorted(result_dir.glob("*.t3D")):
                    stem = _stem_from_t3d(f.name)
                    monitors.append({
                        "stem":      stem,
                        "tree_path": _tree_path(_tree_name_from_t3d(f.name)),
                        "file":      f,
                        "type":      "t3d",
                        "n_steps":   _t3d_step_count(f),
                        "out":       _output_path(cst_file, stem),
                    })
            if monitors:
                results.append((cst_file, result_dir, monitors))
        return results
    else:
        return list(discover_projects(CST_ROOT))


def main():
    if CST_PROJECTS:
        log.info("Single-project mode: %d project(s) listed in CST_PROJECTS", len(CST_PROJECTS))
    else:
        log.info("Full-batch mode: scanning %s", CST_ROOT)
    log.info("")

    projects = _resolve_projects()

    if not projects:
        log.info("No projects with completed field monitors found.")
        return

    total_monitors = sum(len(m) for _, _, m in projects)
    log.info("Found %d project(s) with %d monitor(s) total:", len(projects), total_monitors)
    log.info("")

    for cst_file, result_dir, monitors in projects:
        log.info("  %s", cst_file.name)
        for mon in monitors:
            size_gb = mon["file"].stat().st_size / 1e9
            already = "[exists]" if mon["out"].exists() else ""
            log.info(
                "    [%s] %-40s  %5.1f GB  ->  %s  %s",
                mon["type"].upper(),
                mon["stem"],
                size_gb,
                mon["out"].name,
                already,
            )

    log.info("")

    if DRY_RUN:
        log.info("DRY_RUN=True -- set DRY_RUN=False to actually export.")
        return

    run_export(projects)


if __name__ == "__main__":
    main()
