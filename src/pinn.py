"""Physics-informed neural network (PINN) for the FitzHugh-Nagumo cable equation.

The network maps (x, t) -> (u, v). It is trained on
  * sparse, noisy voltage recordings u(x_k, t) from a few "electrodes",
  * the initial condition and no-flux boundary conditions,
  * the PDE residuals at random collocation points.
The tissue diffusion coefficient D is unknown and learned together with the network.
The recovery variable v is never observed: it is recovered from the physics alone.
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, width=64, depth=5):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


class FHNPinn(nn.Module):
    def __init__(self, params, D_init=0.5, width=64, depth=5):
        super().__init__()
        self.p = params
        self.net = MLP(width=width, depth=depth)
        self.log_D = nn.Parameter(torch.tensor(float(np.log(D_init))))

    @property
    def D(self):
        return torch.exp(self.log_D)

    def forward(self, x, t):
        # scale inputs to [-1, 1]; autograd still differentiates w.r.t. physical x and t
        xs = 2.0 * x / self.p.L - 1.0
        ts = 2.0 * t / self.p.T - 1.0
        out = self.net(torch.cat([xs, ts], dim=1))
        return out[:, :1], out[:, 1:2]

    def pde_residuals(self, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        u, v = self(x, t)
        u_t, u_x = _grad(u, t), _grad(u, x)
        u_xx = _grad(u_x, x)
        v_t = _grad(v, t)
        p = self.p
        r_u = u_t - self.D * u_xx - (u * (1.0 - u) * (u - p.a) - v)
        # divide by eps so both residuals are O(1)
        r_v = (v_t - p.eps * (u - p.gamma * v)) / p.eps
        return r_u, r_v

    def boundary_flux(self, x, t):
        x = x.clone().requires_grad_(True)
        u, _ = self(x, t)
        return _grad(u, x)

    @torch.no_grad()
    def predict_grid(self, x, t, batch=20000):
        """Evaluate u, v on the full (t, x) grid. Returns arrays of shape (nt, nx)."""
        X, Tt = np.meshgrid(x, t)
        xt = torch.tensor(np.stack([X.ravel(), Tt.ravel()], 1), dtype=torch.float32)
        us, vs = [], []
        for i in range(0, len(xt), batch):
            chunk = xt[i:i + batch]
            u, v = self(chunk[:, :1], chunk[:, 1:])
            us.append(u)
            vs.append(v)
        u = torch.cat(us).numpy().reshape(X.shape)
        v = torch.cat(vs).numpy().reshape(X.shape)
        return u, v
