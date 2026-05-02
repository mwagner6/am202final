from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional


@dataclass
class KSParams:
    D_u: float = 1.0
    D_v: float = 1.0
    D_w: float = 1.0
    alpha_u: float = 1.0
    alpha_w: float = 0.0
    beta: float = 1.0

    rho: float = 0.5
    w_max: float = 1.0
    kappa: float = 1.0
    gamma: float = 0.2

    q_type: Literal["classical", "volume_filling", "saturating"] = "volume_filling"
    u_max: float = 1.0
    K: float = 1.0

    Lx: float = 10.0
    Ly: float = 10.0
    Nx: int = 128
    Ny: int = 128

    t_final: float = 20.0
    dt_max: float = 0.01
    save_every: int = 5

    ic_type: Literal[
        "random_bumps",
        "centered_bump",
        "uniform_noise",
        "centered_v_spike",
        "sinusoidal_mode",
        "gaussian_lattice",
        "tumor_blob",
    ] = "random_bumps"
    ic_mean_u: float = 0.5
    ic_mean_v: float = 0.5
    ic_amplitude: float = 0.1
    ic_n_bumps: int = 6
    ic_bump_width: float = 0.8

    ic_v_spike_amplitude: float = 3.0
    ic_v_spike_width: float = 1.0
    ic_u_ring_radius: float = 4.0
    ic_u_ring_thickness: float = 0.8
    ic_u_ring_amplitude: float = 0.3

    ic_mode_nx: int = 4
    ic_mode_ny: int = 0

    ic_lattice_spacing: float = 5.0
    ic_lattice_bump_width: float = 1.2

    ic_tumor_n_blobs: int = 1
    ic_tumor_blob_width: float = 1.5
    ic_tumor_amplitude: float = 0.6
    ic_tumor_zero_outside_radius: float = 0.0

    ic_v_ring_amplitude: float = 0.0
    ic_v_ring_radius: float = 7.0
    ic_v_ring_thickness: float = 1.5

    # None disables the mask; otherwise (x0, x1, y0, y1) restricts initial u
    # to that sub-rectangle of the domain (zero outside).
    ic_u_window: Optional[Tuple[float, float, float, float]] = None

    seed: Optional[int] = 0

    cmap_u: str = "viridis"
    cmap_v: str = "magma"
    fps: int = 30
    figsize: Tuple[float, float] = (10.0, 4.5)

    save_animation: bool = False
    animation_path: str = "chemotaxis.mp4"

    def __post_init__(self):
        if self.q_type not in ("classical", "volume_filling", "saturating"):
            raise ValueError(f"unknown q_type: {self.q_type}")
