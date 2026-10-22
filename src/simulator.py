"""Finite-difference simulation of the 1D FitzHugh-Nagumo cable equation.

    du/dt = D * d2u/dx2 + u (1 - u) (u - a) - v
    dv/dt = eps * (u - gamma * v)

u is the normalised transmembrane voltage and v the slow recovery variable.
The cable has no-flux (Neumann) boundaries and is stimulated at its left end
at t = 0, which launches an action-potential pulse that travels to the right.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class FHNParams:
    D: float = 2.0          # diffusion coefficient (tissue conductivity)
    a: float = 0.1          # excitation threshold
    eps: float = 0.01       # recovery rate
    gamma: float = 0.5      # recovery decay
    L: float = 30.0         # cable length
    T: float = 50.0         # simulated time
    stim_width: float = 4.0 # length of the initially excited region
    stim_edge: float = 0.5  # smoothness of the stimulus edge

    def to_dict(self):
        return asdict(self)


def reaction(u, v, p):
    """Ionic (reaction) terms of the FitzHugh-Nagumo model."""
    f = u * (1.0 - u) * (u - p.a) - v
    g = p.eps * (u - p.gamma * v)
    return f, g


def initial_condition(x, p):
    """Smooth step: the region x < stim_width starts excited, the rest at rest."""
    u0 = 0.5 * (1.0 - np.tanh((x - p.stim_width) / (2.0 * p.stim_edge)))
    v0 = np.zeros_like(x)
    return u0, v0


def simulate(p=None, dx=0.2, dt=0.005, save_every=0.25):
    """Explicit finite-difference solver (2nd-order Laplacian, forward Euler).

    Returns x (nx,), t (nt,), u (nt, nx), v (nt, nx).
    """
    p = p or FHNParams()
    if dt > dx ** 2 / (2.0 * p.D):
        raise ValueError("time step too large for a stable explicit scheme")

    x = np.arange(0.0, p.L + dx / 2, dx)
    u, v = initial_condition(x, p)
    n_steps = int(round(p.T / dt))
    stride = int(round(save_every / dt))

    ts, us, vs = [0.0], [u.copy()], [v.copy()]
    lap = np.empty_like(u)
    for n in range(1, n_steps + 1):
        lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx ** 2
        lap[0] = 2.0 * (u[1] - u[0]) / dx ** 2      # mirrored ghost node => du/dx = 0
        lap[-1] = 2.0 * (u[-2] - u[-1]) / dx ** 2
        f, g = reaction(u, v, p)
        u = u + dt * (p.D * lap + f)
        v = v + dt * g
        if n % stride == 0:
            ts.append(n * dt)
            us.append(u.copy())
            vs.append(v.copy())
    return x, np.array(ts), np.array(us), np.array(vs)


def activation_times(t, u, threshold=0.5):
    """Activation map: first time at which u crosses `threshold` at every x.

    Uses linear interpolation between saved frames; NaN where the tissue never activates.
    """
    act = np.full(u.shape[1], np.nan)
    above = u >= threshold
    for j in range(u.shape[1]):
        idx = np.argmax(above[:, j])
        if not above[idx, j]:
            continue
        if idx == 0:
            act[j] = t[0]
            continue
        u0, u1 = u[idx - 1, j], u[idx, j]
        act[j] = t[idx - 1] + (threshold - u0) / (u1 - u0) * (t[idx] - t[idx - 1])
    return act
