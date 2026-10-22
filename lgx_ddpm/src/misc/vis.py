"""二维点云的可视化工具。"""

from pathlib import Path

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
