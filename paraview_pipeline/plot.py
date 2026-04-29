"""
Matplotlib-based E/H field visualisation.

plot_structured_slice()  — 2-D contour + quiver for rectilinear data
plot_unstructured()      — scatter plot for point-cloud data
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _magnitude(r):
    return np.sqrt(
        np.abs(r.Ex) ** 2 + np.abs(r.Ey) ** 2 + np.abs(r.Ez) ** 2
    ).astype(np.float32)


def _component_map(r, component):
    """Return (data_3d, label) for the requested component."""
    ft = r.field_type
    unit = "V/m" if ft == "E" else "A/m"
    if component == "magnitude":
        return _magnitude(r), f"|{ft}| ({unit})"
    if component == "x":
        return np.abs(r.Ex).astype(np.float32), f"|{ft}_x| ({unit})"
    if component == "y":
        return np.abs(r.Ey).astype(np.float32), f"|{ft}_y| ({unit})"
    if component == "z":
        return np.abs(r.Ez).astype(np.float32), f"|{ft}_z| ({unit})"
    if component == "phase_x":
        d = np.angle(r.Ex.astype(complex), deg=True) if r.is_complex else r.Ex
        return d.astype(np.float32), f"phase({ft}_x) (°)"
    if component == "phase_y":
        d = np.angle(r.Ey.astype(complex), deg=True) if r.is_complex else r.Ey
        return d.astype(np.float32), f"phase({ft}_y) (°)"
    if component == "phase_z":
        d = np.angle(r.Ez.astype(complex), deg=True) if r.is_complex else r.Ez
        return d.astype(np.float32), f"phase({ft}_z) (°)"
    raise ValueError(f"Unknown component '{component}'. "
                     "Choose from: magnitude, x, y, z, phase_x, phase_y, phase_z")


# ---------------------------------------------------------------------------
# Structured slice plot
# ---------------------------------------------------------------------------

def plot_structured_slice(result, axis="z", idx=None, component="magnitude",
                          log_scale=False, output_path=None, show=True,
                          vmin=None, vmax=None, title=None):
    """
    Plot a 2-D cross-section of a StructuredResult.

    Parameters
    ----------
    result     : StructuredResult
    axis       : perpendicular axis of the slice — 'x', 'y', or 'z'
    idx        : grid index along that axis (defaults to midpoint)
    component  : 'magnitude' | 'x' | 'y' | 'z' | 'phase_x' | 'phase_y' | 'phase_z'
    log_scale  : use logarithmic colour normalisation
    output_path: save figure to this path (PNG/PDF/…) if given
    show       : call plt.show()

    Returns
    -------
    fig, ax
    """
    r = result
    data_3d, cbar_label = _component_map(r, component)

    # -- Extract 2-D slice and choose axes -----------------------------------
    # data_3d has shape (nz, ny, nx)
    if axis == "z":
        if idx is None:
            idx = data_3d.shape[0] // 2
        data_2d = data_3d[idx, :, :]           # (ny, nx)
        h_mm = r.x
        v_mm = r.y
        xlabel, ylabel = "x (mm)", "y (mm)"
        title_extra = f"z = {r.z[idx]:.3f} mm"
    elif axis == "y":
        if idx is None:
            idx = data_3d.shape[1] // 2
        data_2d = data_3d[:, idx, :]           # (nz, nx)
        h_mm = r.x
        v_mm = r.z
        xlabel, ylabel = "x (mm)", "z (mm)"
        title_extra = f"y = {r.y[idx]:.3f} mm"
    elif axis == "x":
        if idx is None:
            idx = data_3d.shape[2] // 2
        data_2d = data_3d[:, :, idx]           # (nz, ny)
        h_mm = r.y
        v_mm = r.z
        xlabel, ylabel = "y (mm)", "z (mm)"
        title_extra = f"x = {r.x[idx]:.3f} mm"
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    # -- Build colour norm ---------------------------------------------------
    _vmax = vmax if vmax is not None else float(np.nanmax(np.abs(data_2d)))
    if log_scale and _vmax > 0:
        _vmin = vmin if vmin is not None else max(_vmax * 1e-4, 1e-30)
        norm = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None or vmax is not None) else None

    # -- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(h_mm, v_mm, data_2d, cmap="inferno", norm=norm, shading="auto")
    fig.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else f"{r.field_type}-field {component}  |  {title_extra}")
    ax.set_aspect("equal")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {output_path}")
    if show:
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# Unstructured scatter plot
# ---------------------------------------------------------------------------

def plot_unstructured(result, component="magnitude", log_scale=False,
                      output_path=None, show=True, vmin=None, vmax=None, title=None):
    """
    Scatter plot for unstructured / point-cloud field data.

    Shows x-y and x-z projections side by side.
    """
    r = result
    pos = r.positions   # (N, 3)

    # Re-use _component_map via a lightweight adapter
    class _Adapter:
        Ex = r.Fx; Ey = r.Fy; Ez = r.Fz
        field_type = r.field_type
        is_complex = r.is_complex

    values, cbar_label = _component_map(_Adapter(), component)

    _vmax = vmax if vmax is not None else float(np.nanmax(np.abs(values)))
    if log_scale and _vmax > 0:
        _vmin = vmin if vmin is not None else max(_vmax * 1e-4, 1e-30)
        norm = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None or vmax is not None) else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    kw = dict(c=values, cmap="inferno", s=2, norm=norm, linewidths=0)

    sc1 = ax1.scatter(pos[:, 0] * 1e3, pos[:, 1] * 1e3, **kw)
    fig.colorbar(sc1, ax=ax1, label=cbar_label)
    ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)")
    ax1.set_title("x–y plane"); ax1.set_aspect("equal")

    sc2 = ax2.scatter(pos[:, 0] * 1e3, pos[:, 2] * 1e3, **kw)
    fig.colorbar(sc2, ax=ax2, label=cbar_label)
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("z (mm)")
    ax2.set_title("x–z plane"); ax2.set_aspect("equal")

    fig.suptitle(title if title is not None else f"{r.field_type}-field {component}", fontsize=13)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {output_path}")
    if show:
        plt.show()

    return fig
