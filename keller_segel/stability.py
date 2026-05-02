from dataclasses import dataclass
import math
from .config import KSParams
from .qfuncs import make_q


@dataclass
class StabilityResult:
    is_unstable: bool
    k_plus: float
    k_star: float
    sigma_max: float
    u0: float
    v0: float
    chi: float
    q_u0: float

    def report(self) -> str:
        if not self.is_unstable:
            return (f"uniform state (u0={self.u0:.3f}, v0={self.v0:.3f}) is STABLE: "
                    f"max growth rate {self.sigma_max:+.3e} at any wavenumber.")
        return (
            f"uniform state (u0={self.u0:.3f}, v0={self.v0:.3f}) is UNSTABLE.\n"
            f"  unstable band:    0 < k < k+ = {self.k_plus:.3f}\n"
            f"  most-unstable k:  k* = {self.k_star:.3f}   (wavelength {2*math.pi/self.k_star:.3f})\n"
            f"  growth rate:      sigma(k*) = {self.sigma_max:+.3e}"
        )


def analyze(params: KSParams, u0: float | None = None) -> StabilityResult:
    """Return linear-stability info about the uniform state at u = u0."""
    if u0 is None:
        u0 = params.ic_mean_u
    if params.beta <= 0:
        raise ValueError("Linear stability about a homogeneous state needs beta > 0.")
    v0 = params.alpha_u * u0 / params.beta

    q, qp = make_q(params.q_type, params.u_max, params.K)
    import numpy as np
    q_u0 = float(q(np.array([u0]))[0])
    qp_u0 = float(qp(np.array([u0]))[0])
    chi = q_u0 - qp_u0 * u0

    Du, Dv, alpha, beta = params.D_u, params.D_v, params.alpha_u, params.beta

    rhs = alpha * q_u0 * u0 - Du * chi * beta
    if rhs <= 0 or chi <= 0:
        return StabilityResult(
            is_unstable=False, k_plus=float("nan"), k_star=float("nan"),
            sigma_max=-min(Du * chi, Dv) * 1e-30,
            u0=u0, v0=v0, chi=chi, q_u0=q_u0,
        )

    k_plus_sq = rhs / (Du * chi * Dv)
    k_plus = math.sqrt(k_plus_sq)

    K = np.linspace(1e-4, 3 * k_plus, 4000)
    k2 = K ** 2
    tr = -(Du * chi + Dv) * k2 - beta
    det = k2 * (Du * chi * (Dv * k2 + beta) - alpha * q_u0 * u0)
    disc = tr ** 2 - 4 * det
    sigma = 0.5 * (tr + np.sqrt(np.maximum(disc, 0.0)))
    i_max = int(np.argmax(sigma))
    k_star = float(K[i_max])
    sigma_max = float(sigma[i_max])

    return StabilityResult(
        is_unstable=sigma_max > 0,
        k_plus=k_plus,
        k_star=k_star,
        sigma_max=sigma_max,
        u0=u0, v0=v0, chi=chi, q_u0=q_u0,
    )


def growth_rate(params: KSParams, k: float, u0: float | None = None) -> float:
    """Linear growth rate of a Fourier mode of wavenumber k."""
    if u0 is None:
        u0 = params.ic_mean_u
    if params.beta <= 0:
        raise ValueError("Linear stability needs beta > 0.")
    q, qp = make_q(params.q_type, params.u_max, params.K)
    import numpy as np
    q_u0 = float(q(np.array([u0]))[0])
    qp_u0 = float(qp(np.array([u0]))[0])
    chi = q_u0 - qp_u0 * u0

    Du, Dv, alpha, beta = params.D_u, params.D_v, params.alpha_u, params.beta
    k2 = k * k
    tr = -(Du * chi + Dv) * k2 - beta
    det = k2 * (Du * chi * (Dv * k2 + beta) - alpha * q_u0 * u0)
    disc = tr * tr - 4.0 * det
    if disc < 0:
        return tr / 2.0
    return 0.5 * (tr + math.sqrt(disc))
