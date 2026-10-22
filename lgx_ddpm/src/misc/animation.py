"""将同一批点的真实 DDPM 采样轨迹导出为密度透明度 GIF。"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from .vis import density_opacity


def vis_sampling_gif(
    trajectory: np.ndarray,
    timesteps: np.ndarray,
    *,
    save_path: str | Path,
    fps: int = 20,
    view_quantile: float = 0.995,
    density_bins: int = 80,
    density_gamma: float = 1.0,
) -> Path:
    """保存 GIF，返回路径；trajectory 形状为 (帧数, 点数, 2)。

    timesteps 为每帧实际的扩散时间，从 T 严格递减到 0。每行点的身份在帧间
    保持一致；传入数据应由 inference 的 on_frame 获取，不插值或重采样。
    使用全部帧分位数确定固定视窗，每帧独立计算密度透明度，沿用原坐标。
    20 fps 下普通帧停留 50ms；首帧 600ms、末帧 1500ms，循环播放。
    视窗外点数显示在图中；不修改数据，显示变化不代表模型质量改善。
    """
    points = np.asarray(trajectory)
    times = np.asarray(timesteps)
    if points.ndim != 3 or points.shape[0] < 2 or points.shape[1] < 1 or points.shape[2] != 2 or not np.isfinite(points).all():
        raise ValueError("trajectory 必须为有限的 (F>=2, N>=1, 2) 数组")
    if times.shape != (len(points),) or not np.issubdtype(times.dtype, np.integer) or times[0] <= 0 or times[-1] != 0 or np.any(times[1:] >= times[:-1]):
        raise ValueError("timesteps 必须为从 T>0 严格递减到 0 的整数帧编号")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 100:
        raise ValueError("fps 必须为 1..100 的整数")
    if not 0 < view_quantile <= 1:
        raise ValueError("view_quantile 必须在 (0, 1] 内")
    bound = max(float(np.quantile(np.abs(frame), view_quantile)) for frame in points) + 0.2
    fig = Figure(figsize=(6.4, 6.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.13, top=0.87)
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    fig.suptitle("DDPM | Gaussian to S", fontsize=16, y=0.98)
    fig.text(0.5, 0.025, "Density-weighted opacity | Fixed axes | Original coordinates",
             ha="center", fontsize=9)
    scatter = ax.scatter([], [], s=5, color="#2563eb", edgecolors="none")
    note = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9)
    images = []
    try:
        for frame, t in zip(points, times):
            scatter.set_offsets(frame)
            scatter.set_alpha(density_opacity(frame, bound=bound, bins=density_bins, gamma=density_gamma))
            ax.set_title(f"t = {t} | {times[0] - t} / {times[0]} updates", fontsize=12, pad=12)
            outside = int((np.abs(frame) > bound).any(axis=1).sum())
            note.set_text(f"Outside view: {outside} / {len(frame)}")
            canvas.draw()
            # 逐帧转为调色板图，降低 GIF 编码时的内存占用。
            images.append(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB").quantize(colors=128))
        durations = [max(10, round(100 / fps) * 10)] * len(images)
        durations[0], durations[-1] = 600, 1500
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(destination, save_all=True, append_images=images[1:],
                       duration=durations, loop=0, disposal=2, optimize=False)
        return destination
    finally:
        for image in images:
            image.close()
        fig.clear()
