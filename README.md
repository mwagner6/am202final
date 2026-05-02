# Keller-Segel chemotaxis simulation

A 2D solver for the volume-filling Keller-Segel system

$$u_t = \nabla \cdot \bigl( D_u (q(u) - q'(u) u) \nabla u - q(u) u \nabla v \bigr)$$

$$v_t = D_v \Delta v + \alpha u - \beta v$$

on a periodic box. `u` is the density of motile cells and `v` is the density
of the chemoattractant they follow. There is also a tumor-immune extension
that adds a third field `w` for tumor cells -- see "Tumor extension" below.

## Layout

```
finalproj/
├── run.py                          <-- main 2D run, edit params and run
├── run_tumor.py                    <-- tumor-immune three-field run
├── run_instability_demo.py         <-- Turing instability test (Gaussian lattices)
├── README.md
└── keller_segel/
    ├── __init__.py
    ├── config.py          # KSParams dataclass (all knobs)
    ├── qfuncs.py          # q(u), q'(u)  choices
    ├── initial.py         # initial condition generators
    ├── solver2d.py        # 2D finite-volume / FFT solver
    ├── solver_tumor.py    # 2D three-field tumor-immune solver
    ├── stability.py       # linear stability analysis helpers
    └── viz.py             # matplotlib animation
```

## Usage

```
python run.py              # live animation
python run.py --save       # also write the animation (mp4 or gif)
python run.py --no-show    # headless; pairs with --save
```

All parameters live in the `params = KSParams(...)` block at the top of
[`run.py`](run.py). Edit them and re-run.

## Parameter guide

### PDE coefficients
- `D_u`, `D_v`: diffusivities of cells and chemoattractant
- `alpha`: rate at which cells produce the chemoattractant
- `beta`: decay rate of the chemoattractant

### Motility (`q_type`)

The function $q(u)$ controls how motility depends on local density:

| choice            | $q(u)$                             | behavior                                                   |
|-------------------|------------------------------------|------------------------------------------------------------|
| `classical`       | $1$                                | the original Keller-Segel; can **blow up** in 2D           |
| `volume_filling`  | $\max(1 - u/u_{\max},\, 0)$        | motility shuts off at $u=u_{\max}$; prevents blowup        |
| `saturating`      | $1/(1 + u/K)$                      | smooth saturation                                          |

`volume_filling` is the default and is the safest choice if you want to watch
sharp aggregates form without the integration exploding.

### Grid
- `Nx`, `Ny`: number of cells per axis
- `Lx`, `Ly`: physical box size

### Time integration
- `t_final`: end time
- `dt_max`: fixed timestep
- `save_every`: record a frame every N steps

Both $u$ and $v$ are advanced by forward Euler with central-difference spatial
derivatives on a periodic grid. The timestep must satisfy the diffusion CFL

$$\Delta t \le \frac{1}{2 \max(D_u, D_v) (1/dx^2 + 1/dy^2)}$$

so coarser grids and smaller diffusivities allow larger `dt`.

### Initial conditions (`ic_type`)
- `random_bumps`: a handful of Gaussian blobs at random locations
- `centered_bump`: a single Gaussian at the center of the domain
- `uniform_noise`: spatially uniform $u$ with low-amplitude white noise
- `centered_v_spike`: chemoattractant spike at the center, motile cells in a ring around it
- `sinusoidal_mode`: $u_0 + A\cos(\mathbf{k}\!\cdot\!\mathbf{x})$, for stability tests
- `gaussian_lattice`: lattice of Gaussian bumps (also for stability tests)
- `tumor_blob`: tumor seed in the center, motile cells dispersed (used by `run_tumor.py`)

Other IC controls: `ic_mean_u`, `ic_mean_v`, `ic_amplitude`, `ic_n_bumps`,
`ic_bump_width`, `seed`, plus IC-specific knobs (`ic_v_spike_*`, `ic_u_ring_*`,
`ic_mode_*`, `ic_lattice_*`, `ic_tumor_*`).

### Visualization
- `cmap_u`, `cmap_v`: matplotlib colormaps
- `fps`: animation playback rate

## Notes on the numerics

- Forward Euler in time, central-difference Laplacians and gradients in
  space, periodic boundary conditions.
- Mass of $u$ is conserved exactly on a periodic box (the discrete
  divergence telescopes around the box).
- Central differencing of the chemotactic flux is the simplest choice but
  can produce small oscillations near sharp aggregates: $u$ can undershoot
  below 0 or overshoot above $u_{\max}$. Refine the grid or shrink `dt_max`
  to suppress.

## Getting interesting patterns

To trigger chemotactic aggregation from near-uniform ICs you need the
chemotactic drive to beat diffusion. Try:

- small `D_u` (e.g. 0.1) relative to `D_v` (e.g. 1.0)
- moderate-to-large `alpha` (e.g. 3-5)
- `ic_mean_u` well below `u_max` (so $q(\bar u)$ is not tiny)

The defaults in `run.py` are already tuned for visible pattern formation.

## Tumor extension

`run_tumor.py` solves a three-field tumor-immune system:

$$u_t = \nabla \cdot \bigl(D_u(q-q'u)\nabla u - q\,u\,\nabla v\bigr) - \gamma\,u\,w$$

$$v_t = D_v \Delta v + \alpha\,w - \beta\,v$$

$$w_t = \rho\,w(1 - w/w_{\max}) - \kappa\,u\,w$$

where `u` are motile (immune) cells, `v` is chemoattractant, and `w` is the
tumor cell density. Tumor cells secrete `v` (so motile cells chemotax toward
the tumor), tumor cells grow logistically up to `w_max`, motile cells kill
tumor cells at rate `kappa u w`, and motile cells are consumed by the
encounter at rate `gamma u w`. See [run_tumor.py](run_tumor.py).

## Linear stability demo

`run_instability_demo.py` shows the Turing-like instability of the
homogeneous steady state. It runs two simulations side-by-side, each
seeded with a square lattice of Gaussian bumps:

- one with lattice spacing inside the linearly-unstable wavenumber band
  (bumps grow into a stable pattern)
- one with lattice spacing outside the band (bumps decay back to uniform)

The figure shows live $u$ and $v$ heatmaps for both runs alongside live
log-scale traces of the dominant lattice Fourier amplitude in each field.
