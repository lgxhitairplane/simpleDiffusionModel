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
