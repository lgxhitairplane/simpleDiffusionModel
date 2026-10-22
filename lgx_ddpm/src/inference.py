"""加载训练检查点，生成点云：python -m lgx_ddpm.src.inference --checkpoint .../best.pt。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import sample_target_s
from .model import MLP, inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gif", action="store_true", help="记录真实轨迹并导出密度透明度 GIF")
    parser.add_argument("--frame-every", type=int, default=10, help="每多少次采样更新保存一帧")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if args.frame_every <= 0 or not 1 <= args.fps <= 100:
        parser.error("frame-every 必须为正数，fps 必须在 1..100 内")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["model_t_indexing"] != "0..T-1; model_t = raw_t - 1" or config["normalization"] != "none":
        raise ValueError("检查点的时间步或归一化约定与当前采样器不兼容")
    destination = args.output.resolve() if args.output else checkpoint_path.parent / datetime.now().strftime("samples_%Y%m%d_%H%M%S_%f")
    if destination.exists():
        raise FileExistsError(f"采样输出目录已存在：{destination}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = MLP(**config["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Sampling {args.samples} points on {device}; T={len(checkpoint['beta'])}", flush=True)
    started = time.perf_counter()
    frames, timesteps = [], []

    def record_frame(t, points):
        timesteps.append(t)
        frames.append(points.numpy())

    x0, xt = inference(model, checkpoint["beta"], args.samples, seed=args.seed, return_initial=True,
                       on_frame=record_frame if args.gif else None, frame_every=args.frame_every)
    generated, initial = x0.cpu().numpy(), xt.cpu().numpy()
    elapsed = time.perf_counter() - started
    # 真实目标点仅用于对比，完全不参与反向采样。
    target = sample_target_s(args.samples, noise_std=config["data_metadata"]["noise_std"], seed=args.seed)
    destination.mkdir(parents=True)
    for name, points in (("x0", generated), ("xt", initial), ("target_reference", target)):
        np.save(destination / f"{name}.npy", points, allow_pickle=False)
    metadata = dict(checkpoint=str(checkpoint_path), checkpoint_step=checkpoint["step"],
                    n_samples=args.samples, seed=args.seed, device=device, timesteps=len(checkpoint["beta"]),
                    variance="posterior (beta_tilde)", normalization="none", elapsed_seconds=elapsed,
                    torch_version=str(torch.__version__), numpy_version=np.__version__)
    if args.gif:
        metadata["animation"] = dict(frame_every=args.frame_every, fps=args.fps, frames=len(frames),
                                     view_quantile=0.995, density_bins=80, density_gamma=1.0,
                                     initial_hold_ms=600, final_hold_ms=1500)
    (destination / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from .misc.vis import vis_inference
    fig, _ = vis_inference(initial, generated, target, save_path=destination / "comparison.png")
    fig.clear()
    fig, _ = vis_inference(initial, generated, target, save_path=destination / "comparison_density.png",
                           density_weighted=True, view_quantile=0.995)
    fig.clear()
    print(f"Saved generated X0 and comparison plot: {destination} ({elapsed:.2f}s)", flush=True)
    if args.gif:
        from .misc.animation import vis_sampling_gif
        trajectory, times = np.stack(frames), np.asarray(timesteps, dtype=np.int64)
        np.savez_compressed(destination / "trajectory.npz", points=trajectory, timesteps=times)
        print(f"Rendering {len(times)} real sampling frames...", flush=True)
        path = vis_sampling_gif(trajectory, times, save_path=destination / "sampling_density.gif", fps=args.fps)
        print(f"Saved GIF: {path}", flush=True)


if __name__ == "__main__":
    main()
