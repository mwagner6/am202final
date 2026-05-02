import argparse
import time

from keller_segel import KSParams, SolverTumorNoDrift2D, initial, viz


params = KSParams(
    D_u=0.1,
    D_v=1.0,
    D_w=0.1,
    alpha_u=0.25,
    alpha_w=2.0,
    beta=0.5,
    q_type="volume_filling",
    u_max=1.5,
    K=1.0,

    rho=0.5,
    w_max=1.0,
    kappa=1.5,
    gamma=0.3,

    Lx=20.0, Ly=20.0,
    Nx=64, Ny=64,

    t_final=40.0,
    dt=0.005,
    save_every=40,

    ic_type="tumor_blob",
    ic_mean_u=0.2,
    ic_amplitude=0.05,
    ic_u_ring_radius=7.0,
    ic_u_ring_thickness=1.5,
    ic_u_ring_amplitude=0.6,
    ic_tumor_n_blobs=1,
    ic_tumor_blob_width=2.0,
    ic_tumor_amplitude=1.0,
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
            params.animation_path = "tumor_nodrift.mp4"

    print("== Tumor-immune chemotaxis simulation ==")
    print(f"  grid:       {params.Nx} x {params.Ny}    domain: {params.Lx} x {params.Ly}")
    print(f"  immune:     D_u={params.D_u}  alpha_u={params.alpha_u}  beta={params.beta}  "
          f"gamma={params.gamma}")
    print(f"  tumor src:  alpha_w={params.alpha_w}")
    print(f"  tumor:      rho={params.rho}  w_max={params.w_max}  kappa={params.kappa}")
    print(f"  initial u:  uniform mean {params.ic_mean_u}  /  initial w: blob amp {params.ic_tumor_amplitude}")
    print(f"  t_final = {params.t_final}, dt = {params.dt}")
    print()

    u0, v0, w0 = initial.make_initial_tumor(params, ndim=2)
    solver = SolverTumorNoDrift2D(params, u0, v0, w0)

    t0 = time.perf_counter()
    frames = solver.run(progress=True)
    print(f"\nintegration done: {solver.step_count} steps, "
          f"{len(frames)} frames, {time.perf_counter() - t0:.2f}s wall")

    viz.save_tumor_snapshots(frames, params, "tumor_nodrift_snapshots.png",
                              times=(0.0, 10.0, 40.0))
    viz.save_tumor_population_curves(frames, params,
                                       "tumor_nodrift_population.png")

    anim = viz.animate_tumor_2d(frames, params, show=not args.no_show)
    return anim


if __name__ == "__main__":
    main()
