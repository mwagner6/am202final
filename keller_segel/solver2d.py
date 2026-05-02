import numpy as np
from .config import KSParams


class Solver2D:
    def __init__(self, params: KSParams, u0: np.ndarray, v0: np.ndarray):
        self.p = params
        self.u = u0.astype(np.float64).copy()
        self.v = v0.astype(np.float64).copy()
        self.dx = params.Lx / params.Nx
        self.dy = params.Ly / params.Ny
        self.t = 0.0
        self.step_count = 0

    def laplacian(self, a):
        return ((np.roll(a, -1, 0) - 2 * a + np.roll(a, 1, 0)) / self.dx ** 2 +
                (np.roll(a, -1, 1) - 2 * a + np.roll(a, 1, 1)) / self.dy ** 2)

    def grad(self, a):
        gx = (np.roll(a, -1, 0) - np.roll(a, 1, 0)) / (2 * self.dx)
        gy = (np.roll(a, -1, 1) - np.roll(a, 1, 1)) / (2 * self.dy)
        return gx, gy

    def rhs(self, u, v):
        p = self.p
        q = 1.0 - u / p.u_max

        gvx, gvy = self.grad(v)
        Fx = -q * u * gvx
        Fy = -q * u * gvy

        divF = ((np.roll(Fx, -1, 0) - np.roll(Fx, 1, 0)) / (2 * self.dx) +
                (np.roll(Fy, -1, 1) - np.roll(Fy, 1, 1)) / (2 * self.dy))

        du_dt = p.D_u * self.laplacian(u) + divF
        dv_dt = p.D_v * self.laplacian(v) + p.alpha_u * u - p.beta * v
        return du_dt, dv_dt

    def step(self, dt):
        du_dt, dv_dt = self.rhs(self.u, self.v)
        self.u = self.u + dt * du_dt
        self.v = self.v + dt * dv_dt
        self.t += dt
        self.step_count += 1
        return dt

    def run(self, progress=True):
        p = self.p
        dt = p.dt
        frames = [(self.t, self.u.copy(), self.v.copy())]
        while self.t < p.t_final - 1e-12:
            self.step(dt)
            if self.step_count % p.save_every == 0:
                frames.append((self.t, self.u.copy(), self.v.copy()))
                if progress:
                    print(f"  t = {self.t:7.3f}/{p.t_final:.3f}  dt = {dt:.3e}  "
                          f"u:[{self.u.min():.3f},{self.u.max():.3f}]  "
                          f"v:[{self.v.min():.3f},{self.v.max():.3f}]")
        frames.append((self.t, self.u.copy(), self.v.copy()))
        return frames
