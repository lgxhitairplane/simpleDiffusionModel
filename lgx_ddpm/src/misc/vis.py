"""二维点云的可视化工具。"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def vis_s_distribution(
    s_distribution: np.ndarray,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """绘制 S 形分布样本，保持横纵坐标的单位长度一致。

    Args:
        s_distribution: 形状为 (N, 2) 的有限实数坐标，例如 target_S 的返回值。
        save_path: 可选图片保存路径；自动创建父目录。
        show: 是否显示图像。批量生成图片时可设为 False。

    Returns:
        (fig, ax)，可用于进一步修改图像。使用完后可调用 plt.close(fig)。
    """
    points = np.asarray(s_distribution)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("s_distribution 必须是非空的 (N, 2) 坐标数组")
    if not np.issubdtype(points.dtype, np.number) or not np.isrealobj(points):
        raise ValueError("s_distribution 必须包含实数坐标")
    if not np.isfinite(points).all():
        raise ValueError("s_distribution 不能包含 NaN 或无穷大")

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(
        points[:, 0], points[:, 1], s=5, alpha=0.45,
        color="#2563eb", edgecolors="none",
    )
    ax.set(title="S-shaped target distribution", xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    return fig, ax


def group_training_data(
    rawdata_dir: str | Path,
    timesteps: tuple[int, ...] = (100, 300, 500, 700, 1000),
    chunk_size: int = 65536,
) -> dict[int, dict[str, np.ndarray]]:
    """按精确相等的 t 分组，合并不同批次中的同时间步样本。

    仅将所选时间步的数据载入内存；每组的 Xt、X0、epsilon 保持行对应。
    不重采样、不重新加噪，也不通过 Xt 反推 X0。
    """
    root = Path(rawdata_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("rawdata 尚未完整生成")
    if not timesteps or any(
        isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
        or not 1 <= t <= metadata["timesteps"] for t in timesteps
    ):
        raise ValueError("timesteps 必须是 1..T 范围内的整数")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    groups = {int(t): {name: [] for name in ("xt", "x0", "epsilon")} for t in timesteps}
    arrays = {}
    try:
        for name in ("t", "xt", "x0", "epsilon"):
            arrays[name] = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            expected = (metadata["n_samples"],) if name == "t" else (metadata["n_samples"], 2)
            if arrays[name].shape != expected:
                raise ValueError(f"{name}.npy 的形状与 metadata.json 不一致")
        for start in range(0, metadata["n_samples"], chunk_size):
            stop = min(start + chunk_size, metadata["n_samples"])
            batch_t = arrays["t"][start:stop]
            for t, group in groups.items():
                mask = batch_t == t
                if mask.any():
                    for name in group:
                        group[name].append(arrays[name][start:stop][mask])
    finally:
        for array in arrays.values():
            array._mmap.close()
    return {
        t: {name: np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.float32)
            for name, parts in group.items()}
        for t, group in groups.items()
    }


def vis_training_data(
    rawdata_dir: str | Path,
    timesteps: tuple[int, ...] = (100, 300, 500, 700, 1000),
    *,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """每个 t 保存一张 Xt / X0 / epsilon 三联图，返回图片路径。

    五组图使用统一的等比例坐标轴；同一行样本在三幅子图中颜色相同。
    不同 t 组来自不同的随机 X0，这些图展示分布而非同一批点的轨迹。
    """
    root = Path(rawdata_dir)
    groups = group_training_data(root, timesteps)
    alpha_bar = np.cumprod(1.0 - np.load(root / "beta.npy", allow_pickle=False))
    destination = Path(output_dir) if output_dir is not None else root / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    bound = max(
        (float(np.abs(points).max()) for group in groups.values()
         for points in group.values() if len(points)), default=3.0,
    )
    limit = np.ceil((bound + 0.2) * 2) / 2
    paths = []
    for t, group in groups.items():
        count = len(group["xt"])
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True, sharey=True)
        colors = group["x0"][:, 1]
        for ax, name, title in zip(
            axes, ("xt", "x0", "epsilon"),
            ("Xt | noisy training input", "X0 | original S samples", "epsilon | noise target"),
        ):
            points = group[name]
            ax.scatter(points[:, 0], points[:, 1], c=colors, cmap="viridis",
                       vmin=-2.2, vmax=2.2, s=7, alpha=0.65, edgecolors="none")
            ax.set(title=title, xlabel="x", xlim=(-limit, limit), ylim=(-limit, limit))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
            if not count:
                ax.text(0.5, 0.5, "No samples at this timestep", ha="center", transform=ax.transAxes)
        axes[0].set_ylabel("y")
        fig.suptitle(f"t = {t}  |  n = {count:,}  |  alpha_bar = {alpha_bar[t-1]:.6f}", fontsize=14)
        fig.text(0.5, 0.025,
                 "Same samples and colors across panels. Noise target is unscaled N(0, I).",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.15, 1, 0.92))
        path = destination / f"t_{t:04d}.png"
        try:
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)
        paths.append(path)
    return paths


def density_opacity(points, *, bound: float, bins: int = 80, gamma: float = 1.0):
    """把局部点密度归一化为透明度，返回每个点的 alpha，范围为 0.015..0.65。

    在 [-bound, bound]^2 上建立二维直方图，累加相邻 3x3 格计数，
    按点所在格读取密度，再除以可见点密度的第 95 百分位并截断到 [0,1]。
    alpha = 0.015 + 0.635 * normalized_density**gamma。
    gamma 越大，稀疏点越淡；每组点云独立归一化，不借助目标曲线判断点的好坏。
    超出视窗的点使用最低透明度；不删除点、不修改坐标，仅改变显示权重。
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("points 必须是非空的有限 (N, 2) 数组")
    if not np.isfinite(bound) or bound <= 0 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("bound、gamma 必须为有限正数")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 3:
        raise ValueError("bins 必须是至少为 3 的整数")
    visible = (np.abs(points) <= bound).all(axis=1)
    opacity = np.full(len(points), 0.015)
    if not visible.any():
        return opacity
    selected = points[visible]
    counts, _, _ = np.histogram2d(selected[:, 0], selected[:, 1], bins=bins,
                                 range=((-bound, bound), (-bound, bound)))
    padded = np.pad(counts, 1)
    local_counts = sum(padded[i:i + bins, j:j + bins] for i in range(3) for j in range(3))
    cells = np.floor((selected + bound) / (2 * bound) * bins).astype(np.int64)
    cells = np.clip(cells, 0, bins - 1)
    density = local_counts[cells[:, 0], cells[:, 1]]
    normalized = np.clip(density / np.quantile(density, 0.95), 0, 1)
    opacity[visible] += 0.635 * normalized ** gamma
    return opacity


def vis_inference(initial, generated, target, *, save_path: str | Path,
                  density_weighted: bool = False, view_quantile: float = 1.0,
                  density_bins: int = 80, density_gamma: float = 1.0):
    """对比初始、生成及目标点云；可选密度透明度和分位数视窗。

    默认保留原始散点显示。density_weighted=True 淡化稀疏点；
    view_quantile=0.995 以各组绝对坐标的 99.5% 分位数确定共同范围，
    避免极端点撑大画布。仍使用原坐标和等比例坐标轴，图中标注视窗外点数。
    这是可视化调整，不代表生成分布质量提高；不修改传入数据。
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    arrays = [np.asarray(points) for points in (initial, generated, target)]
    for points in arrays:
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            raise ValueError("每组点云必须为非空的有限 (N, 2) 数组")
    fig = Figure(figsize=(13, 5.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.84, wspace=0.08)
    if not 0 < view_quantile <= 1:
        raise ValueError("view_quantile 必须在 (0, 1] 内")
    bound = max(float(np.quantile(np.abs(points), view_quantile)) for points in arrays) + 0.2
    for ax, points, title in zip(axes, arrays, ("Initial XT | N(0, I)", "Generated X0 | DDPM", "Target | S distribution")):
        alpha = density_opacity(points, bound=bound, bins=density_bins, gamma=density_gamma) if density_weighted else 0.45
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=alpha, edgecolors="none")
        ax.set(title=title, xlabel="x", xlim=(-bound, bound), ylim=(-bound, bound))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        if view_quantile < 1:
            outside = int((np.abs(points) > bound).any(axis=1).sum())
            ax.text(0.02, 0.02, f"Outside view: {outside} / {len(points)}", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel("y")
    if density_weighted:
        fig.suptitle(f"Density-weighted opacity (gamma={density_gamma:g}) | Coordinates unchanged", fontsize=12)
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    return fig, axes


def vis_loss(
    run_dir: str | Path,
    *,
    save_path: str | Path | None = None,
    smoothing_window: int = 100,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """读取训练记录，绘制逐步 loss、后向滑动平均和验证集 loss。

    横轴为优化器更新次数，不是扩散时间步 t。默认保存 run_dir/loss.png。
    平滑仅用于显示，不修改原始 loss；验证曲线使用真实评估步数（含初始 0）。
    默认使用无窗口画布，适用于训练结束后自动保存；返回 (fig, ax)。
    """
    root = Path(run_dir)
    losses = np.load(root / "train_loss.npy", allow_pickle=False)
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    if losses.ndim != 1 or not losses.size or not np.isfinite(losses).all():
        raise ValueError("train_loss.npy 必须为非空的有限一维数组")
    if isinstance(smoothing_window, bool) or not isinstance(smoothing_window, int) or smoothing_window <= 0:
        raise ValueError("smoothing_window 必须为正整数")
    val_steps = np.asarray([item["step"] for item in history], dtype=np.int64)
    val_losses = np.asarray([item["validation_loss"] for item in history], dtype=np.float64)
    if not len(val_losses) or not np.isfinite(val_losses).all():
        raise ValueError("history.json 必须包含有限的验证 loss")
    if np.any(np.diff(val_steps) <= 0) or val_steps[0] < 0 or val_steps[-1] > len(losses):
        raise ValueError("验证步数必须递增，并与训练记录范围一致")

    if show:
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(10, 5.5), constrained_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
    steps = np.arange(1, len(losses) + 1)
    window = min(smoothing_window, len(losses))
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(steps, losses, color="#93c5fd", alpha=0.5, linewidth=0.7, label="Training MSE (each batch)")
    ax.plot(steps[window - 1:], smoothed, color="#2563eb", linewidth=1.8,
            label=f"Training MSE ({window}-step trailing mean)")
    ax.plot(val_steps, val_losses, "o-", color="#ea580c", markersize=4, linewidth=1.6,
            label="Validation MSE")
    best = int(np.argmin(val_losses))
    ax.scatter(val_steps[best], val_losses[best], marker="*", s=160, color="#b91c1c",
               zorder=5, label=f"Best validation: {val_losses[best]:.4f} at step {val_steps[best]}")
    ax.set(title="DDPM noise prediction loss", xlabel="Training update step", ylabel="Mean squared error")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max(1, len(losses)))
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    destination = Path(save_path) if save_path is not None else root / "loss.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    if show:
        plt.show()
    return fig, ax
