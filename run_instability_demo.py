import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from keller_segel import KSParams, Solver2D, initial, stability


BASE = KSParams(
    D_u=0.05, D_v=1.0,
    alpha_u=2.0, beta=1.0,
    q_type="volume_filling", u_max=1.0,
    Lx=20.0, Ly=20.0, Nx=64, Ny=64,
    t_final=20.0,
    dt=0.005,
    save_every=80,
    ic_type="gaussian_lattice",
    ic_mean_u=0.3,
    ic_amplitude=0.04,
    seed=0,
    cmap_u="viridis",
    cmap_v="magma",
    fps=30,
)

A_UNSTABLE = 5.0
A_STABLE   = 2.0
WIDTH_FACTOR = 0.25


def run_one(spacing: float, label: str):
    p = KSParams(**{**BASE.__dict__})
    p.ic_lattice_spacing = spacing
    p.ic_lattice_bump_width = WIDTH_FACTOR * spacing
    k_a = 2 * math.pi / spacing
    sigma_pred = stability.growth_rate(p, k_a)
    n_bumps = int(round((p.Lx / spacing) * (p.Ly / spacing)))
    print(f"--- {label} run:  a = {spacing}  ({n_bumps} bumps total)  ->  "
          f"k_a = {k_a:.3f}  ->  sigma = {sigma_pred:+.4f}")
    print(f"    (e-fold time {1/abs(sigma_pred):.2f}; linear amplification by t={p.t_final}: "
          f"e^{sigma_pred * p.t_final:+.2f} = {math.exp(sigma_pred * p.t_final):.3g})")

    u0, v0 = initial.make_initial(p, ndim=2)
    solver = Solver2D(p, u0, v0)
    frames = solver.run(progress=False)
    return p, frames, k_a, sigma_pred


def lattice_amplitude(field, spacing, Lx, Ly):
    """Dominant lattice Fourier amplitude of a 2D field."""
    Nx, Ny = field.shape
    nx_lat = int(round(Lx / spacing))
    ny_lat = int(round(Ly / spacing))
    F = np.fft.fft2(field - field.mean()) / (Nx * Ny)
    return (abs(F[nx_lat, ny_lat])
            + abs(F[-nx_lat, ny_lat])
            + abs(F[nx_lat, -ny_lat])
            + abs(F[-nx_lat, -ny_lat]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="save the animation to instability_demo.mp4 (or .gif)")
    ap.add_argument("--no-show", action="store_true",
                    help="skip the matplotlib window")
    args = ap.parse_args()

    print()
    print(stability.analyze(BASE).report())
    print()

    t0 = time.perf_counter()
    p_un, frames_un, k_un, sig_un = run_one(A_UNSTABLE, "UNSTABLE")
    p_st, frames_st, k_st, sig_st = run_one(A_STABLE,   "STABLE  ")
    print(f"both runs done in {time.perf_counter()-t0:.2f}s wall")
    print()

    times_un = np.array([f[0] for f in frames_un])
    amps_u_un = np.array([lattice_amplitude(f[1], A_UNSTABLE, p_un.Lx, p_un.Ly) for f in frames_un])
    amps_v_un = np.array([lattice_amplitude(f[2], A_UNSTABLE, p_un.Lx, p_un.Ly) for f in frames_un])

    times_st = np.array([f[0] for f in frames_st])
    amps_u_st = np.array([lattice_amplitude(f[1], A_STABLE, p_st.Lx, p_st.Ly) for f in frames_st])
    amps_v_st = np.array([lattice_amplitude(f[2], A_STABLE, p_st.Lx, p_st.Ly) for f in frames_st])

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.30)

    ax_u_un = fig.add_subplot(gs[0, 0])
    ax_u_st = fig.add_subplot(gs[0, 1])
    ax_u_amp = fig.add_subplot(gs[0, 2])
    ax_v_un = fig.add_subplot(gs[1, 0])
    ax_v_st = fig.add_subplot(gs[1, 1])
    ax_v_amp = fig.add_subplot(gs[1, 2])

    extent = (0, BASE.Lx, 0, BASE.Ly)

    all_u = np.concatenate([f[1].ravel() for f in frames_un] +
                            [f[1].ravel() for f in frames_st])
    all_v = np.concatenate([f[2].ravel() for f in frames_un] +
                            [f[2].ravel() for f in frames_st])
    u_vmin, u_vmax = float(np.percentile(all_u, 2.0)), float(np.percentile(all_u, 99.5))
    v_vmin, v_vmax = float(np.percentile(all_v, 2.0)), float(np.percentile(all_v, 99.5))

    im_u_un = ax_u_un.imshow(frames_un[0][1].T, origin="lower", extent=extent,
                              cmap=BASE.cmap_u, vmin=u_vmin, vmax=u_vmax,
                              aspect="equal", interpolation="nearest")
    im_u_st = ax_u_st.imshow(frames_st[0][1].T, origin="lower", extent=extent,
                              cmap=BASE.cmap_u, vmin=u_vmin, vmax=u_vmax,
                              aspect="equal", interpolation="nearest")
    im_v_un = ax_v_un.imshow(frames_un[0][2].T, origin="lower", extent=extent,
                              cmap=BASE.cmap_v, vmin=v_vmin, vmax=v_vmax,
                              aspect="equal", interpolation="nearest")
    im_v_st = ax_v_st.imshow(frames_st[0][2].T, origin="lower", extent=extent,
                              cmap=BASE.cmap_v, vmin=v_vmin, vmax=v_vmax,
                              aspect="equal", interpolation="nearest")

    ax_u_un.set_title(fr"$u(x,y,t)$  $a={A_UNSTABLE}$  (in band)")
    ax_u_st.set_title(fr"$u(x,y,t)$  $a={A_STABLE}$  (out of band)")
    ax_v_un.set_title(fr"$v(x,y,t)$  $a={A_UNSTABLE}$  (in band)")
    ax_v_st.set_title(fr"$v(x,y,t)$  $a={A_STABLE}$  (out of band)")
    for ax in (ax_u_un, ax_u_st, ax_v_un, ax_v_st):
        ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.colorbar(im_u_st, ax=ax_u_st, fraction=0.046, pad=0.04)
    fig.colorbar(im_v_st, ax=ax_v_st, fraction=0.046, pad=0.04)

    line_u_un, = ax_u_amp.semilogy(times_un[:1], amps_u_un[:1], "-",
                                    color="tomato",   label=fr"$a={A_UNSTABLE}$ (in band)")
    line_u_st, = ax_u_amp.semilogy(times_st[:1], amps_u_st[:1], "-",
                                    color="steelblue", label=fr"$a={A_STABLE}$ (out of band)")
    ax_u_amp.set_xlim(0, BASE.t_final)
    ax_u_amp.set_ylim(min(amps_u_un.min(), amps_u_st.min()) * 0.5,
                       max(amps_u_un.max(), amps_u_st.max()) * 2.0)
    ax_u_amp.set_xlabel("time")
    ax_u_amp.set_title("u amplitude (live)")
    ax_u_amp.legend(fontsize=8, loc="lower right")
    ax_u_amp.grid(True, which="both", alpha=0.3)

    # v starts uniform (zero spatial structure), so floor below to avoid log(0)
    floor = 1e-10
    amps_v_un_plot = np.maximum(amps_v_un, floor)
    amps_v_st_plot = np.maximum(amps_v_st, floor)

    line_v_un, = ax_v_amp.semilogy(times_un[:1], amps_v_un_plot[:1], "-",
                                    color="tomato",   label=fr"$a={A_UNSTABLE}$ (in band)")
    line_v_st, = ax_v_amp.semilogy(times_st[:1], amps_v_st_plot[:1], "-",
                                    color="steelblue", label=fr"$a={A_STABLE}$ (out of band)")
    ax_v_amp.set_xlim(0, BASE.t_final)
    v_ymin = max(min(amps_v_un_plot[amps_v_un_plot > floor].min() if (amps_v_un_plot > floor).any() else floor,
                      amps_v_st_plot[amps_v_st_plot > floor].min() if (amps_v_st_plot > floor).any() else floor) * 0.5,
                  floor)
    v_ymax = max(amps_v_un_plot.max(), amps_v_st_plot.max()) * 2.0
    ax_v_amp.set_ylim(v_ymin, v_ymax)
    ax_v_amp.set_xlabel("time")
    ax_v_amp.set_ylabel("v lattice amplitude")
    ax_v_amp.set_title("v amplitude (live)")
    ax_v_amp.legend(fontsize=8, loc="lower right")
    ax_v_amp.grid(True, which="both", alpha=0.3)

    title = fig.suptitle(f"t = {frames_un[0][0]:.2f}", fontsize=14, y=0.995)
    n_frames = min(len(frames_un), len(frames_st))

    def update(i):
        t_un, u_un, v_un = frames_un[i]
        t_st, u_st, v_st = frames_st[i]
        im_u_un.set_data(u_un.T)
        im_u_st.set_data(u_st.T)
        im_v_un.set_data(v_un.T)
        im_v_st.set_data(v_st.T)
        line_u_un.set_data(times_un[:i+1], amps_u_un[:i+1])
        line_u_st.set_data(times_st[:i+1], amps_u_st[:i+1])
        line_v_un.set_data(times_un[:i+1], amps_v_un_plot[:i+1])
        line_v_st.set_data(times_st[:i+1], amps_v_st_plot[:i+1])
        title.set_text(f"t = {t_un:.2f}")
        return (im_u_un, im_u_st, im_v_un, im_v_st,
                line_u_un, line_u_st, line_v_un, line_v_st, title)

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / BASE.fps, blit=False)
    

    if args.save:
        path = "instability_demo.mp4"
        print(f"writing animation -> {path}")
        try:
            anim.save(path, fps=BASE.fps, dpi=110)
        except Exception as e:
            fallback = "instability_demo.gif"
            print(f"  mp4 write failed ({e}); writing {fallback}")
            anim.save(fallback, fps=BASE.fps, dpi=100, writer="pillow")


    if not args.no_show:
        plt.show()
    return anim


if __name__ == "__main__":
    main()
