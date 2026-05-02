import argparse
import time

from keller_segel import KSParams, SolverTumor2D, initial, viz


# Show only the central 20x20 region of the 40x40 simulation domain.
DISPLAY_WINDOW = (10.0, 30.0, 10.0, 30.0)


params = KSParams(
    D_u=0.05,
    D_v=0.5,
    D_w=0.1,
    alpha_u=0.25,
    alpha_w=0.5,
    beta=0.05,
    q_type="volume_filling",
    u_max=1.0,
    K=1.0,

    rho=0.5,
    w_max=1.0,
    kappa=1.0,
    gamma=0.2,

    Lx=40.0, Ly=40.0,
    Nx=128, Ny=128,

    t_final=30.0,
    dt_max=0.005,
    save_every=40,

    ic_type="tumor_blob",
    ic_mean_u=0.2,
    ic_amplitude=0.05,
    ic_u_ring_radius=7.0,
    ic_u_ring_thickness=1.5,
    ic_u_ring_amplitude=0.6,
    ic_u_window=(10.0, 30.0, 10.0, 30.0),
    ic_tumor_n_blobs=1,
    ic_tumor_blob_width=2.0,
    ic_tumor_amplitude=1.0,
    ic_tumor_zero_outside_radius=6.0,

    ic_v_ring_amplitude=5.0,
    ic_v_ring_radius=10.0,
    ic_v_ring_thickness=1.5,

    seed=0,

    cmap_u="viridis",
    cmap_v="magma",
    fps=30,
    figsize=(13.0, 6.5),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="save the animation to params.animation_path")
    ap.add_argument("--no-show", action="store_true",
                    help="skip the matplotlib window")
    args = ap.parse_args()

    if args.save:
        params.save_animation = True
        if params.animation_path == "chemotaxis.mp4":
            params.animation_path = "tumor_misdirected.mp4"

    print("== Tumor-immune simulation with misleading initial v ring ==")
    print(f"  grid:       {params.Nx} x {params.Ny}    domain: {params.Lx} x {params.Ly}")
    print(f"  immune:     D_u={params.D_u}  alpha_u={params.alpha_u}  beta={params.beta}  "
          f"gamma={params.gamma}")
    print(f"  tumor src:  alpha_w={params.alpha_w}")
    print(f"  tumor:      rho={params.rho}  w_max={params.w_max}  kappa={params.kappa}")
    print(f"  v ring:     amplitude {params.ic_v_ring_amplitude} at radius {params.ic_v_ring_radius}")
    print(f"  t_final = {params.t_final}, dt = {params.dt_max}")
    print()

    u0, v0, w0 = initial.make_initial_tumor(params, ndim=2)
    solver = SolverTumor2D(params, u0, v0, w0)

    t0 = time.perf_counter()
    frames = solver.run(progress=True)
    print(f"\nintegration done: {solver.step_count} steps, "
          f"{len(frames)} frames, {time.perf_counter() - t0:.2f}s wall")

    viz.save_tumor_snapshots(frames, params, "tumor_misdirected_snapshots.png",
                              times=(0.0, 10.0, 40.0),
                              display_window=DISPLAY_WINDOW)
    viz.save_tumor_population_curves(frames, params,
                                       "tumor_misdirected_population.png")

    anim = viz.animate_tumor_2d(frames, params, show=not args.no_show,
                                  display_window=DISPLAY_WINDOW)
    return anim


if __name__ == "__main__":
    main()
