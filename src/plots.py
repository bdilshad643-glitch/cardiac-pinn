"""Figures for the README."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _field(ax, x, t, z, title, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(z, origin="lower", aspect="auto", extent=[x[0], x[-1], t[0], t[-1]],
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("position x")
    ax.set_ylabel("time t")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def plot_fields(x, t, u_true, u_pinn, v_true, v_pinn, electrodes, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    lo, hi = u_true.min(), u_true.max()
    _field(axes[0, 0], x, t, u_true, "voltage u: simulation", lo, hi)
    for e in electrodes:
        axes[0, 0].axvline(e, color="w", ls="--", lw=0.8)
    _field(axes[0, 1], x, t, u_pinn, "voltage u: PINN", lo, hi)
    _field(axes[0, 2], x, t, np.abs(u_pinn - u_true), "|error| in u", cmap="magma")
    lo, hi = v_true.min(), v_true.max()
    _field(axes[1, 0], x, t, v_true, "recovery v: simulation", lo, hi)
    _field(axes[1, 1], x, t, v_pinn, "recovery v: PINN (never observed)", lo, hi)
    _field(axes[1, 2], x, t, np.abs(v_pinn - v_true), "|error| in v", cmap="magma")
    fig.suptitle("FitzHugh-Nagumo wave: dashed lines mark the electrodes the PINN sees")
    fig.savefig(path)
    plt.close(fig)


def plot_activation_map(x, act_true, act_pinn, electrodes, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    ax.plot(x, act_true, "k-", lw=2, label="simulation")
    ax.plot(x, act_pinn, "C1--", lw=2, label="PINN")
    for i, e in enumerate(electrodes):
        ax.axvline(e, color="0.7", ls=":", lw=1, label="electrode" if i == 0 else None)
    ax.set_xlabel("position x")
    ax.set_ylabel("activation time")
    ax.set_title("Activation map (time the voltage first crosses 0.5)")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_traces(x, t, u_true, u_pinn, electrodes, u_obs, path):
    """Voltage over time at one electrode and at a location between electrodes."""
    e = electrodes[1]
    between = 0.5 * (electrodes[2] + electrodes[3])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True, sharey=True)
    j = np.argmin(np.abs(x - e))
    axes[0].plot(t, u_obs[:, 1], ".", color="0.6", ms=3, label="noisy recording")
    axes[0].plot(t, u_true[:, j], "k-", lw=1.5, label="simulation")
    axes[0].plot(t, u_pinn[:, j], "C1--", lw=1.5, label="PINN")
    axes[0].set_title(f"at electrode x = {e:.1f}")
    j = np.argmin(np.abs(x - between))
    axes[1].plot(t, u_true[:, j], "k-", lw=1.5, label="simulation")
    axes[1].plot(t, u_pinn[:, j], "C1--", lw=1.5, label="PINN")
    axes[1].set_title(f"between electrodes, x = {x[j]:.1f} (no data)")
    for ax in axes:
        ax.set_xlabel("time t")
        ax.legend(frameon=False)
    axes[0].set_ylabel("voltage u")
    fig.savefig(path)
    plt.close(fig)


def plot_training(history, true_D, path, switch=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    it = history["iter"]
    axes[0].semilogy(it, history["loss"], label="total")
    axes[0].semilogy(it, history["data"], label="electrode data")
    axes[0].semilogy(it, history["pde"], label="PDE residual")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("training loss")
    axes[0].legend(frameon=False)
    axes[1].plot(it, history["D"], "C1", label="learned D")
    axes[1].axhline(true_D, color="k", ls="--", label=f"true D = {true_D}")
    axes[1].set_xlabel("iteration")
    axes[1].set_title("unknown diffusion coefficient")
    if switch:
        for ax in axes:
            ax.axvline(switch, color="0.6", ls=":", lw=1)
            ax.text(switch, 0.98, " L-BFGS", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="0.4")
    axes[1].legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)
