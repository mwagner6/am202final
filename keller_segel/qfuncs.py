import numpy as np


def make_q(q_type: str, u_max: float = 1.0, K: float = 1.0):
    """Return (q, q_prime) callables for the chosen motility law."""
    if q_type == "classical":
        def q(u):
            return np.ones_like(u)

        def q_prime(u):
            return np.zeros_like(u)

    elif q_type == "volume_filling":
        def q(u):
            return np.maximum(1.0 - u / u_max, 0.0)

        def q_prime(u):
            out = np.where(u < u_max, -1.0 / u_max, 0.0)
            return out

    elif q_type == "saturating":
        def q(u):
            return 1.0 / (1.0 + u / K)

        def q_prime(u):
            return -(1.0 / K) / (1.0 + u / K) ** 2

    else:
        raise ValueError(f"unknown q_type: {q_type}")

    return q, q_prime
