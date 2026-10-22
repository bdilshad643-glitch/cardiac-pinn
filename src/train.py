"""Reconstruct a cardiac activation wave from sparse electrode recordings with a PINN.

    python src/train.py                                  # full run (results/ is overwritten)
    python src/train.py --iters 2000 --lbfgs-steps 0     # quick smoke test

Training has two phases: Adam on resampled collocation points, then L-BFGS with a
stronger physics weight, which pins down the unknown diffusion coefficient D.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pinn import FHNPinn
from plots import plot_activation_map, plot_fields, plot_traces, plot_training
from simulator import FHNParams, activation_times, initial_condition, simulate

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=20000, help="Adam iterations")
    ap.add_argument("--lbfgs-steps", type=int, default=60, help="L-BFGS steps (20 inner iterations each)")
    ap.add_argument("--n-electrodes", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.01, help="std of Gaussian noise on recordings")
    ap.add_argument("--n-colloc", type=int, default=8000, help="PDE collocation points (resampled)")
    ap.add_argument("--D-init", type=float, default=0.5, help="initial guess for the unknown D")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    return ap.parse_args()


def tensor(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32).reshape(-1, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 1. ground truth from the finite-difference solver ------------------------
    p = FHNParams()
    x, t, u_true, v_true = simulate(p)
    print(f"simulated {len(t)} x {len(x)} grid, true D = {p.D}")

    # ---- 2. sparse, noisy electrode recordings (the only voltage data the PINN sees) ----
    electrodes = (np.arange(args.n_electrodes) + 0.5) * p.L / args.n_electrodes
    e_idx = np.array([np.argmin(np.abs(x - e)) for e in electrodes])
    electrodes = x[e_idx]
    T_obs, X_obs = np.meshgrid(t, electrodes, indexing="ij")
    u_obs = u_true[:, e_idx] + args.noise * rng.standard_normal(T_obs.shape)
    x_d, t_d, u_d = tensor(X_obs.ravel()), tensor(T_obs.ravel()), tensor(u_obs.ravel())

    # initial and boundary points
    x_ic = np.linspace(0, p.L, 300)
    u0, v0 = initial_condition(x_ic, p)
    x_ic, t_ic, u_ic, v_ic = tensor(x_ic), tensor(np.zeros_like(x_ic)), tensor(u0), tensor(v0)
    t_b = np.linspace(0, p.T, 200)
    x_bc = tensor(np.concatenate([np.zeros_like(t_b), np.full_like(t_b, p.L)]))
    t_bc = tensor(np.concatenate([t_b, t_b]))

    def collocation():
        return tensor(rng.uniform(0, p.L, args.n_colloc)), tensor(rng.uniform(0, p.T, args.n_colloc))

    # ---- 3. train -------------------------------------------------------------------
    model = FHNPinn(p, D_init=args.D_init)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1 / args.iters))  # 2e-3 -> 2e-5
    mse = lambda r: torch.mean(r ** 2)

    def losses(x_c, t_c):
        u_pred, _ = model(x_d, t_d)
        u_0, v_0 = model(x_ic, t_ic)
        r_u, r_v = model.pde_residuals(x_c, t_c)
        return {"data": mse(u_pred - u_d),
                "ic": mse(u_0 - u_ic) + mse(v_0 - v_ic),
                "bc": mse(model.boundary_flux(x_bc, t_bc)),
                "pde": mse(r_u) + mse(r_v)}

    def total(l, w_pde):
        return 10.0 * l["data"] + 10.0 * l["ic"] + l["bc"] + w_pde * l["pde"]

    def log(it, loss, l):
        history["iter"].append(it)
        history["loss"].append(loss.item())
        history["data"].append(l["data"].item())
        history["pde"].append(l["pde"].item())
        history["D"].append(model.D.item())

    history = {"iter": [], "loss": [], "data": [], "pde": [], "D": []}
    x_c, t_c = collocation()
    start = time.time()
    for it in range(1, args.iters + 1):
        if it % 500 == 0:
            x_c, t_c = collocation()
        opt.zero_grad()
        l = losses(x_c, t_c)
        loss = total(l, w_pde=1.0)
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == 1:
            log(it, loss, l)
        if it % 1000 == 0 or it == 1:
            print(f"adam {it:6d}  loss {loss.item():.3e}  data {l['data'].item():.2e}  "
                  f"pde {l['pde'].item():.2e}  D {model.D.item():.3f}  ({time.time() - start:.0f}s)", flush=True)

    # L-BFGS refinement on a larger, fixed set of collocation points with a 10x physics weight
    x_c, t_c = tensor(rng.uniform(0, p.L, 20000)), tensor(rng.uniform(0, p.T, 20000))
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = losses(x_c, t_c)
        loss = total(l, w_pde=10.0)
        loss.backward()
        closure.last = (loss, l)
        return loss

    for step in range(1, args.lbfgs_steps + 1):
        lbfgs.step(closure)
        it = args.iters + 20 * step
        log(it, *closure.last)
        if step % 10 == 0:
            loss, l = closure.last
            print(f"lbfgs {it:5d}  loss {loss.item():.3e}  data {l['data'].item():.2e}  "
                  f"pde {l['pde'].item():.2e}  D {model.D.item():.3f}  ({time.time() - start:.0f}s)", flush=True)

    # ---- 4. evaluate on the full grid ----------------------------------------------------
    u_pinn, v_pinn = model.predict_grid(x, t)
    rel_l2 = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    act_true = activation_times(t, u_true)
    act_pinn = activation_times(t, u_pinn)
    both = ~np.isnan(act_true) & ~np.isnan(act_pinn)
    D_hat = model.D.item()
    metrics = {
        "true_D": p.D,
        "estimated_D": round(D_hat, 4),
        "D_error_percent": round(100 * abs(D_hat - p.D) / p.D, 2),
        "u_relative_L2_error_percent": round(100 * rel_l2(u_pinn, u_true), 2),
        "v_relative_L2_error_percent": round(100 * rel_l2(v_pinn, v_true), 2),
        "activation_time_MAE": round(float(np.mean(np.abs(act_pinn[both] - act_true[both]))), 3),
        "activation_time_max_error": round(float(np.max(np.abs(act_pinn[both] - act_true[both]))), 3),
        "activated_points_matched": f"{int(both.sum())}/{int((~np.isnan(act_true)).sum())}",
        "n_electrodes": args.n_electrodes,
        "electrode_positions": [round(float(e), 2) for e in electrodes],
        "observed_fraction_of_grid_percent": round(100 * len(e_idx) / len(x), 2),
        "noise_std": args.noise,
        "adam_iterations": args.iters,
        "lbfgs_iterations": 20 * args.lbfgs_steps,
        "train_seconds": round(time.time() - start, 1),
        "seed": args.seed,
        "fhn_params": p.to_dict(),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # ---- 5. figures ------------------------------------------------------------------------
    plot_fields(x, t, u_true, u_pinn, v_true, v_pinn, electrodes, args.out / "fields.png")
    plot_activation_map(x, act_true, act_pinn, electrodes, args.out / "activation_map.png")
    plot_traces(x, t, u_true, u_pinn, electrodes, u_obs, args.out / "traces.png")
    plot_training(history, p.D, args.out / "training.png", switch=args.iters if args.lbfgs_steps else None)
    torch.save(model.state_dict(), args.out / "pinn.pt")


if __name__ == "__main__":
    main()
