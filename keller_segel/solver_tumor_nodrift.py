import numpy as np
from .config import KSParams


class SolverTumorNoDrift2D:
    def __init__(self, params: KSParams,
                 u0: np.ndarray, v0: np.ndarray, w0: np.ndarray):
        self.p = params
        self.u = u0.astype(np.float64).copy()
        self.v = v0.astype(np.float64).copy()
        self.w = w0.astype(np.float64).copy()
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

    def rhs(self, u, v, w):
        p = self.p
        du_dt = p.D_u * self.laplacian(u) - p.gamma * u * w
        dv_dt = p.D_v * self.laplacian(v) + p.alpha_u * u + p.alpha_w * w - p.beta * v
        dw_dt = p.D_w * self.laplacian(w) + p.rho * w * (1.0 - w / p.w_max) - p.kappa * u * w
        return du_dt, dv_dt, dw_dt

    def step(self, dt):
        du_dt, dv_dt, dw_dt = self.rhs(self.u, self.v, self.w)
        self.u = self.u + dt * du_dt
        self.v = self.v + dt * dv_dt
        self.w = self.w + dt * dw_dt
        self.t += dt
        self.step_count += 1
        return dt

    def run(self, progress=True):
        p = self.p
        dt = p.dt
        frames = [(self.t, self.u.copy(), self.v.copy(), self.w.copy())]
        while self.t < p.t_final - 1e-12:
            self.step(dt)
            if self.step_count % p.save_every == 0:
                frames.append((self.t, self.u.copy(), self.v.copy(), self.w.copy()))
                if progress:
                    print(f"  t={self.t:7.3f}  dt={dt:.2e}  "
                          f"u:[{self.u.min():.3f},{self.u.max():.3f}]  "
                          f"v:[{self.v.min():.3f},{self.v.max():.3f}]  "
                          f"w:[{self.w.min():.3f},{self.w.max():.3f}]  "
                          f"tumor mass={self.w.sum()*self.dx*self.dy:.3f}")
        frames.append((self.t, self.u.copy(), self.v.copy(), self.w.copy()))
        return frames
