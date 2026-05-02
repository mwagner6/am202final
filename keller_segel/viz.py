import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .config import KSParams


def _contrast_limits(arrays, lo_pct=2.0, hi_pct=99.0):
    """Percentile-based (vmin, vmax) pooled across a sequence of arrays."""
    flat = np.concatenate([a.ravel() for a in arrays])
    vmin = float(np.percentile(flat, lo_pct))
    vmax = float(np.percentile(flat, hi_pct))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    return vmin, vmax


def _crop_field(field, params: KSParams, display_window):
    """Crop a 2D field to (x0, x1, y0, y1); return (array, extent)."""
    if display_window is None:
        return field, (0.0, params.Lx, 0.0, params.Ly)
    x0, x1, y0, y1 = display_window
    dx = params.Lx / params.Nx
    dy = params.Ly / params.Ny
    i0 = max(0, int(round(x0 / dx)))
    i1 = min(params.Nx, int(round(x1 / dx)))
    j0 = max(0, int(round(y0 / dy)))
    j1 = min(params.Ny, int(round(y1 / dy)))
    return field[i0:i1, j0:j1], (i0 * dx, i1 * dx, j0 * dy, j1 * dy)


def save_tumor_snapshots(frames, params: KSParams, path: str,
                         times=(0.0, 10.0, 40.0), cmap_w="cividis",
                         display_window=None):
    """Save a 3x3 PNG (rows=times, cols=u/v/w) for the tumor-immune system."""
    frame_times = np.array([f[0] for f in frames])
    selected = []
    for t_target in times:
        i = int(np.argmin(np.abs(frame_times - t_target)))
        selected.append(frames[i])

    cropped = []
    extent = (0.0, params.Lx, 0.0, params.Ly)
    for frame in selected:
        t = frame[0]
        cu, extent = _crop_field(frame[1], params, display_window)
        cv, _      = _crop_field(frame[2], params, display_window)
        cw, _      = _crop_field(frame[3], params, display_window)
        cropped.append((t, cu, cv, cw))

    u_vmin, u_vmax = _contrast_limits([f[1] for f in cropped])
    v_vmin, v_vmax = _contrast_limits([f[2] for f in cropped])
    w_vmin, w_vmax = _contrast_limits([f[3] for f in cropped])

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.0),
                              constrained_layout=True)

    field_specs = [
        (1, "u  (motile cells)",    params.cmap_u, u_vmin, u_vmax),
        (2, "v  (chemoattractant)", params.cmap_v, v_vmin, v_vmax),
        (3, "w  (tumor cells)",     cmap_w,        w_vmin, w_vmax),
    ]

    last_ims = [None, None, None]
    for row, frame in enumerate(cropped):
        t = frame[0]
        for col, (idx, title, cmap, vmin, vmax) in enumerate(field_specs):
            ax = axes[row, col]
            im = ax.imshow(frame[idx].T, origin="lower", extent=extent,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            aspect="equal", interpolation="nearest")
            last_ims[col] = im
            if row == 0:
                ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        axes[row, 0].annotate(f"t = {t:.1f}",
                                xy=(-0.28, 0.5), xycoords="axes fraction",
                                ha="center", va="center",
                                rotation=90, fontsize=14, fontweight="bold")

    for col in range(3):
        fig.colorbar(last_ims[col], ax=axes[:, col].tolist(),
                      shrink=0.85, pad=0.02, location="right")

    fig.suptitle(f"Snapshots at t = {', '.join(f'{t:.1f}' for t in times)}",
                  fontsize=14)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote snapshot grid -> {path}")


def save_tumor_population_curves(frames, params: KSParams, path: str):
    """Save a PNG of total motile-cell mass and total tumor mass over time."""
    cell_area = (params.Lx / params.Nx) * (params.Ly / params.Ny)
    times  = np.array([f[0] for f in frames])
    mass_u = np.array([f[1].sum() * cell_area for f in frames])
    mass_w = np.array([f[3].sum() * cell_area for f in frames])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(times, mass_u, "-", color="tab:green", linewidth=2,
             label="motile cells (u)")
    ax.plot(times, mass_w, "-", color="tab:red", linewidth=2,
             label="tumor cells (w)")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, max(mass_u.max(), mass_w.max()) * 1.15)
    ax.set_xlabel("time")
    ax.set_ylabel("total mass (integrated over the domain)")
    ax.set_title("Population dynamics: total cell mass over time")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    ax.annotate(f"u(final) = {mass_u[-1]:.2f}",
                 xy=(times[-1], mass_u[-1]), xytext=(-100, 8),
                 textcoords="offset points", fontsize=9, color="tab:green")
    ax.annotate(f"w(final) = {mass_w[-1]:.2f}",
                 xy=(times[-1], mass_w[-1]), xytext=(-100, 8),
                 textcoords="offset points", fontsize=9, color="tab:red")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote population curves -> {path}")


def animate_2d(frames, params: KSParams, show=True):
    """Animate the two-field Keller-Segel system."""
    fig, axes = plt.subplots(1, 2, figsize=params.figsize)

    u_vmin, u_vmax = _contrast_limits([u for _, u, _ in frames])
    v_vmin, v_vmax = _contrast_limits([v for _, _, v in frames])

    extent = (0.0, params.Lx, 0.0, params.Ly)

    im_u = axes[0].imshow(frames[0][1].T, origin="lower", extent=extent,
                          cmap=params.cmap_u, vmin=u_vmin, vmax=u_vmax,
                          aspect="equal", interpolation="nearest")
    im_v = axes[1].imshow(frames[0][2].T, origin="lower", extent=extent,
                          cmap=params.cmap_v, vmin=v_vmin, vmax=v_vmax,
                          aspect="equal", interpolation="nearest")

    axes[0].set_title("u  (cell density)")
    axes[1].set_title("v  (chemoattractant)")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im_u, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im_v, ax=axes[1], fraction=0.046, pad=0.04)

    title = fig.suptitle(f"t = {frames[0][0]:.3f}")

    def update(i):
        t, u, v = frames[i]
        im_u.set_data(u.T)
        im_v.set_data(v.T)
        title.set_text(f"t = {t:.3f}")
        return im_u, im_v, title

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / params.fps, blit=False)

    if params.save_animation:
        _save_anim(anim, params)

    if show:
        plt.show()
    return anim


def animate_tumor_2d(frames, params: KSParams, show=True, cmap_w="cividis",
                      display_window=None):
    """Animate the three-field tumor-immune system."""
    fig = plt.figure(figsize=(13.0, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1.0], hspace=0.45, wspace=0.35)
    ax_u = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[0, 2])
    ax_t = fig.add_subplot(gs[1, :])

    def crop(field):
        cropped, _ = _crop_field(field, params, display_window)
        return cropped
    _, extent = _crop_field(frames[0][1], params, display_window)

    u_vmin, u_vmax = _contrast_limits([crop(f[1]) for f in frames])
    v_vmin, v_vmax = _contrast_limits([crop(f[2]) for f in frames])
    w_vmin, w_vmax = _contrast_limits([crop(f[3]) for f in frames])

    im_u = ax_u.imshow(crop(frames[0][1]).T, origin="lower", extent=extent,
                       cmap=params.cmap_u, vmin=u_vmin, vmax=u_vmax,
                       aspect="equal", interpolation="nearest")
    im_v = ax_v.imshow(crop(frames[0][2]).T, origin="lower", extent=extent,
                       cmap=params.cmap_v, vmin=v_vmin, vmax=v_vmax,
                       aspect="equal", interpolation="nearest")
    im_w = ax_w.imshow(crop(frames[0][3]).T, origin="lower", extent=extent,
                       cmap=cmap_w, vmin=w_vmin, vmax=w_vmax,
                       aspect="equal", interpolation="nearest")

    ax_u.set_title("u  (motile cells)")
    ax_v.set_title("v  (chemoattractant)")
    ax_w.set_title("w  (tumor cells)")
    for ax in (ax_u, ax_v, ax_w):
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
    fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
    fig.colorbar(im_w, ax=ax_w, fraction=0.046, pad=0.04)

    cell_area = (params.Lx / params.Nx) * (params.Ly / params.Ny)
    times = np.array([f[0] for f in frames])
    mass_u = np.array([f[1].sum() * cell_area for f in frames])
    mass_w = np.array([f[3].sum() * cell_area for f in frames])

    line_u, = ax_t.plot(times[:1], mass_u[:1], "-", color="tab:green", label="motile cells (u)")
    line_w, = ax_t.plot(times[:1], mass_w[:1], "-", color="tab:red",   label="tumor cells (w)")
    ax_t.set_xlim(0, times[-1])
    ax_t.set_ylim(0, max(mass_u.max(), mass_w.max()) * 1.15)
    ax_t.set_xlabel("time"); ax_t.set_ylabel("total mass")
    ax_t.set_title("Total cell populations over time")
    ax_t.legend(loc="best", fontsize=9)
    ax_t.grid(alpha=0.3)

    title = fig.suptitle(f"t = {frames[0][0]:.3f}", y=0.995)

    def update(i):
        t, u, v, w = frames[i]
        im_u.set_data(crop(u).T); im_v.set_data(crop(v).T); im_w.set_data(crop(w).T)
        line_u.set_data(times[:i+1], mass_u[:i+1])
        line_w.set_data(times[:i+1], mass_w[:i+1])
        title.set_text(f"t = {t:.3f}")
        return im_u, im_v, im_w, line_u, line_w, title

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / params.fps, blit=False)

    if params.save_animation:
        _save_anim(anim, params)
    if show:
        plt.show()
    return anim


def _save_anim(anim, params: KSParams):
    path = params.animation_path
    print(f"  writing animation -> {path}")
    try:
        anim.save(path, fps=params.fps, dpi=120)
    except Exception as e:
        # fall back to gif if ffmpeg is missing
        print(f"  mp4 write failed ({e}); falling back to gif")
        fallback = path.rsplit(".", 1)[0] + ".gif"
        anim.save(fallback, fps=params.fps, dpi=100, writer="pillow")
        print(f"  wrote {fallback}")
