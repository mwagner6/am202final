import argparse
import time

from keller_segel import KSParams, Solver2D, initial, viz


params = KSParams(
    D_u=0.02,
    D_v=0.3,
    alpha_u=0.0,
    beta=0.05,

    q_type="volume_filling",
    u_max=1.0,
    K=1.0,

    Lx=20.0, Ly=20.0,
    Nx=128, Ny=128,

    t_final=40.0,
    dt_max=0.01,
    save_every=15,

    ic_type="centered_v_spike",
    ic_mean_u=0.0,
    ic_mean_v=0.0,
    ic_amplitude=0.05,
    ic_v_spike_amplitude=2.5,
    ic_v_spike_width=6.0,
    ic_u_ring_radius=5.0,
    ic_u_ring_thickness=1.0,
    ic_u_ring_amplitude=0.6,
    seed=0,

    cmap_u="viridis",
    cmap_v="magma",
    fps=30,
    figsize=(11.0, 4.8),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="save the animation to params.animation_path")
    ap.add_argument("--no-show", action="store_true",
                    help="don't pop up the matplotlib window (pairs with --save)")
    args = ap.parse_args()

    if args.save:
        params.save_animation = True

    print("== Keller-Segel 2D simulation ==")
    print(f"  grid:   {params.Nx} x {params.Ny}   domain: {params.Lx} x {params.Ly}")
    print(f"  q(u):   {params.q_type}   D_u={params.D_u}  D_v={params.D_v}  "
          f"alpha_u={params.alpha_u}  beta={params.beta}")
    print(f"  t_final = {params.t_final}, dt = {params.dt_max}")

    u0, v0 = initial.make_initial(params, ndim=2)
    solver = Solver2D(params, u0, v0)

    t0 = time.perf_counter()
    frames = solver.run(progress=True)
    print(f"integration done: {solver.step_count} steps, "
          f"{len(frames)} frames, {time.perf_counter() - t0:.2f}s wall")

    anim = viz.animate_2d(frames, params, show=not args.no_show)
    return anim


if __name__ == "__main__":
    main()
