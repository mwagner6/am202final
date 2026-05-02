import numpy as np
from .config import KSParams


def make_initial_tumor(params: KSParams, ndim: int = 2):
    """Return (u0, v0, w0) for the three-field tumor-immune system."""
    if ndim != 2:
        raise NotImplementedError("only ndim=2 is supported")

    rng = np.random.default_rng(params.seed)
    xs = np.linspace(0, params.Lx, params.Nx, endpoint=False)
    ys = np.linspace(0, params.Ly, params.Ny, endpoint=False)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    shape = X.shape

    cx_c, cy_c = params.Lx / 2, params.Ly / 2
    dxc = X - cx_c
    dxc = dxc - params.Lx * np.round(dxc / params.Lx)
    dyc = Y - cy_c
    dyc = dyc - params.Ly * np.round(dyc / params.Ly)
    r = np.sqrt(dxc ** 2 + dyc ** 2)

    u = np.full(shape, params.ic_mean_u, dtype=np.float64)
    if params.ic_u_ring_amplitude > 0:
        ring = np.exp(-((r - params.ic_u_ring_radius) ** 2)
                       / (2 * params.ic_u_ring_thickness ** 2))
        u = u + params.ic_u_ring_amplitude * ring
    u = u + params.ic_amplitude * rng.standard_normal(shape) * 0.1
    u = np.clip(u, 0.0, None)

    if params.ic_u_window is not None:
        x0, x1, y0, y1 = params.ic_u_window
        inside = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
        u = np.where(inside, u, 0.0)

    v = np.zeros(shape, dtype=np.float64)
    if getattr(params, "ic_v_ring_amplitude", 0.0) > 0:
        v_ring = np.exp(-((r - params.ic_v_ring_radius) ** 2)
                         / (2 * params.ic_v_ring_thickness ** 2))
        v = v + params.ic_v_ring_amplitude * v_ring

    w = np.zeros(shape, dtype=np.float64)
    n = params.ic_tumor_n_blobs
    sig = params.ic_tumor_blob_width

    if n == 1:
        centers = [(cx_c, cy_c)]
    else:
        centers = [(rng.uniform(0.35 * params.Lx, 0.65 * params.Lx),
                    rng.uniform(0.35 * params.Ly, 0.65 * params.Ly))
                   for _ in range(n)]

    cutoff = params.ic_tumor_zero_outside_radius
    for cx, cy in centers:
        dx = X - cx
        dx = dx - params.Lx * np.round(dx / params.Lx)
        dy = Y - cy
        dy = dy - params.Ly * np.round(dy / params.Ly)
        bump = params.ic_tumor_amplitude * np.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))
        if cutoff > 0:
            r_to_blob = np.sqrt(dx ** 2 + dy ** 2)
            bump = np.where(r_to_blob > cutoff, 0.0, bump)
        w = w + bump

    w = np.clip(w, 0.0, params.w_max)
    return u, v, w


def make_initial(params: KSParams, ndim: int = 2):
    """Return (u0, v0) for the two-field Keller-Segel system."""
    if ndim != 2:
        raise NotImplementedError("only ndim=2 is supported")
    rng = np.random.default_rng(params.seed)

    shape = (params.Nx, params.Ny)
    xs = np.linspace(0, params.Lx, params.Nx, endpoint=False)
    ys = np.linspace(0, params.Ly, params.Ny, endpoint=False)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    centers_L = (params.Lx, params.Ly)
    coords = (X, Y)

    u = np.full(shape, params.ic_mean_u, dtype=np.float64)
    v = np.full(shape, params.ic_mean_v, dtype=np.float64)

    if params.ic_type == "gaussian_lattice":
        a = params.ic_lattice_spacing
        sig = params.ic_lattice_bump_width

        if abs(params.Lx % a) > 1e-9 or abs(params.Ly % a) > 1e-9:
            raise ValueError(
                f"ic_lattice_spacing={a} must divide Lx={params.Lx} and Ly={params.Ly} evenly "
                f"so the lattice fits the periodic box without seams."
            )

        X, Y = coords
        bump_field = np.zeros_like(X)
        n_x = int(round(params.Lx / a))
        n_y = int(round(params.Ly / a))
        for i in range(n_x):
            for j in range(n_y):
                cx = (i + 0.5) * a
                cy = (j + 0.5) * a
                dx = X - cx
                dx = dx - params.Lx * np.round(dx / params.Lx)
                dy = Y - cy
                dy = dy - params.Ly * np.round(dy / params.Ly)
                bump_field += np.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))

        # zero-mean perturbation rescaled to ic_amplitude peak
        bump_field = bump_field - bump_field.mean()
        bf_max = float(np.abs(bump_field).max())
        if bf_max > 0:
            bump_field = bump_field * (params.ic_amplitude / bf_max)

        u = params.ic_mean_u + bump_field
        if params.beta > 0:
            v = (params.alpha_u / params.beta) * params.ic_mean_u * np.ones_like(u)
        else:
            v = params.ic_mean_v * np.ones_like(u)
        u = np.clip(u, 0.0, None)
        v = np.clip(v, 0.0, None)
        return u, v

    if params.ic_type == "sinusoidal_mode":
        u0_mean = params.ic_mean_u
        phase = 2 * np.pi * (params.ic_mode_nx * coords[0] / params.Lx
                              + params.ic_mode_ny * coords[1] / params.Ly)
        u = u0_mean + params.ic_amplitude * np.cos(phase)
        if params.beta > 0:
            v = (params.alpha_u / params.beta) * u0_mean * np.ones_like(u)
        else:
            v = params.ic_mean_v * np.ones_like(u)
        u = u + 1e-4 * rng.standard_normal(shape)
        v = np.clip(v, 0.0, None)
        u = np.clip(u, 0.0, None)
        return u, v

    if params.ic_type == "uniform_noise":
        u += params.ic_amplitude * rng.standard_normal(shape)

    elif params.ic_type == "centered_bump":
        center = tuple(L / 2 for L in centers_L)
        r2 = sum((c - cc) ** 2 for c, cc in zip(coords, center))
        u += params.ic_amplitude * np.exp(-r2 / (2 * params.ic_bump_width ** 2))

    elif params.ic_type == "centered_v_spike":
        center = tuple(L / 2 for L in centers_L)
        r2 = 0.0
        for c, cc, L in zip(coords, center, centers_L):
            d = c - cc
            d = d - L * np.round(d / L)
            r2 = r2 + d ** 2
        r = np.sqrt(r2)

        v = v + params.ic_v_spike_amplitude * np.exp(
            -r2 / (2 * params.ic_v_spike_width ** 2)
        )

        ring = np.exp(
            -((r - params.ic_u_ring_radius) ** 2) / (2 * params.ic_u_ring_thickness ** 2)
        )
        u = u + params.ic_u_ring_amplitude * ring
        u = u + 0.02 * params.ic_amplitude * rng.standard_normal(shape)

    elif params.ic_type == "random_bumps":
        for _ in range(params.ic_n_bumps):
            center = tuple(rng.uniform(0, L) for L in centers_L)
            r2 = 0.0
            for c, cc, L in zip(coords, center, centers_L):
                d = c - cc
                d = d - L * np.round(d / L)
                r2 = r2 + d ** 2
            amp = params.ic_amplitude * rng.uniform(0.5, 1.5)
            u += amp * np.exp(-r2 / (2 * params.ic_bump_width ** 2))
        u += 0.02 * params.ic_amplitude * rng.standard_normal(shape)

    else:
        raise ValueError(f"unknown ic_type: {params.ic_type}")

    u = np.clip(u, 0.0, None)
    if params.q_type == "volume_filling":
        u = np.clip(u, 0.0, 0.95 * params.u_max)

    v = np.clip(v, 0.0, None)
    return u, v
