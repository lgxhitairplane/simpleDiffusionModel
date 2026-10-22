"""Offline figures and a GIF of one actual reverse diffusion trajectory."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def render(output, target, trajectory, steps, losses, animate=True):
    plt.rcParams.update({"figure.facecolor": "#f8fafc", "axes.facecolor": "#f8fafc",
                         "axes.spines.top": False, "axes.spines.right": False})
    bound = max(3.5, float(np.abs(target).max()) + 0.3,
                float(np.abs(trajectory[[0, -1]]).max()) + 0.2)

    def setup(ax, title):
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), title=title, xlabel="x", ylabel="y")
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)

    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(trajectory[0])))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.15)
    for ax, values, title in zip(axes, [target, trajectory[0], trajectory[-1]],
                                 ["Target S distribution", "Initial Gaussian noise", "Generated samples"]):
        ax.scatter(*values.T, s=3, alpha=0.5, color="#2563eb")
        setup(ax, title)
    fig.savefig(output / "comparison.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.plot(np.arange(1, len(losses) + 1), losses, alpha=0.25, linewidth=0.6, label="Batch MSE")
    window = min(100, len(losses))
    if window:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(losses) + 1), smooth, label=f"Mean over {window} steps")
    ax.set(xlabel="Training update", ylabel="Noise prediction MSE", title="Training loss")
    ax.legend()
    fig.savefig(output / "loss.png", dpi=160)
    plt.close(fig)

    if not animate:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.15)
    axes[0].scatter(*target.T, s=3, color="#94a3b8", alpha=0.4)
    setup(axes[0], "Target S distribution")
    cloud = axes[1].scatter(*trajectory[0].T, s=4, c=colors, alpha=0.65)
    setup(axes[1], "Reverse diffusion")
    # Each point retains its color/index throughout the same stochastic trajectory.
    tracked = np.arange(min(12, trajectory.shape[1]))
    trails = [axes[1].plot([], [], color=colors[i], alpha=0.5, lw=0.6)[0] for i in tracked]

    def update(frame):
        cloud.set_offsets(trajectory[frame])
        axes[1].set_title(f"Reverse diffusion | remaining steps: {steps[frame]}")
        for i, line in zip(tracked, trails):
            line.set_data(trajectory[max(0, frame - 8):frame + 1, i, 0],
                          trajectory[max(0, frame - 8):frame + 1, i, 1])
        return [cloud, *trails]

    animation = FuncAnimation(fig, update, frames=len(trajectory), interval=70, blit=False)
    animation.save(output / "sampling.gif", writer=PillowWriter(fps=15), dpi=100)
    plt.close(fig)
