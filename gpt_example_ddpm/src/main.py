"""Train and visualize tiny-diffusion on 2D S points, with CUDA support."""
import argparse
import json
import os
from pathlib import Path
import random
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "temp" / "matplotlib"))

import numpy as np
import torch
from torch.nn import functional as F
from tiny_diffusion import MLP, NoiseScheduler
from data import make_s_curve, sliced_wasserstein
from visualize import render


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--steps", type=positive, default=5000, help="optimizer updates")
    p.add_argument("--batch-size", type=positive, default=512)
    p.add_argument("--data-size", type=positive, default=20000)
    p.add_argument("--samples", type=positive, default=2000)
    p.add_argument("--timesteps", type=positive, default=1000, help="diffusion chain length")
    p.add_argument("--hidden-size", type=positive, default=128)
    p.add_argument("--hidden-layers", type=positive, default=3)
    p.add_argument("--embedding-size", type=positive, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.04, help="S-curve thickness before normalization")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=positive, default=101)
    p.add_argument("--run-name", default=None)
    p.add_argument("--checkpoint", type=Path, help="sample only from an existing checkpoint")
    args = p.parse_args()
    if args.data_size < 2 or args.timesteps < 2 or args.frames < 2:
        p.error("data-size, timesteps and frames must be >= 2")
    if args.embedding_size < 4 or args.embedding_size % 2:
        p.error("embedding-size must be even and >= 4")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.noise) or args.noise < 0:
        p.error("lr must be finite and positive; noise must be finite and nonnegative")
    if args.run_name and (args.run_name in (".", "..") or any(c in args.run_name for c in '/\\:')):
        p.error("run-name must be a single directory name")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def sample(model, scheduler, count, frames, device):
    model.eval()
    x = torch.randn(count, 2, device=device)
    snapshots, remaining = [x.cpu().numpy().copy()], [scheduler.num_timesteps]
    keep = set(np.linspace(0, scheduler.num_timesteps, min(frames, scheduler.num_timesteps + 1), dtype=int))
    for t in range(scheduler.num_timesteps - 1, -1, -1):
        times = torch.full((count,), t, device=device, dtype=torch.long)
        x = scheduler.step(model(x, times), t, x)
        if t in keep:
            snapshots.append(x.cpu().numpy().copy())
            remaining.append(t)
    trajectory = np.stack(snapshots)
    if not np.isfinite(trajectory).all():
        raise RuntimeError("Non-finite samples; inspect training and the noise schedule")
    return trajectory, np.asarray(remaining)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install CUDA PyTorch or use --device cpu.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed_all(args.seed)
    if device.type == "cuda":
        # Exercise a real CUDA kernel, not just driver discovery.
        torch.ones(2, 2, device=device).square().sum().item()
    print(f"Device: {device} | {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    output = ROOT / "Data" / "runs" / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["checkpoint"] = str(args.checkpoint.resolve()) if args.checkpoint else None
    config["device_used"] = str(device)
    losses = []
    start = time.perf_counter()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model_config = checkpoint["model_config"]
        scheduler = NoiseScheduler(checkpoint["timesteps"]).to(device)
        model = MLP(**model_config).to(device)
        model.load_state_dict(checkpoint["model"])
        data_config = checkpoint["data_config"]
        losses = checkpoint["losses"]
        config["source_training_config"] = checkpoint["config"]
    else:
        model_config = {"hidden_size": args.hidden_size, "hidden_layers": args.hidden_layers,
                        "emb_size": args.embedding_size}
        model = MLP(**model_config).to(device)
        scheduler = NoiseScheduler(args.timesteps).to(device)
        data_config = {"count": args.data_size, "noise": args.noise, "seed": args.seed}
    points, parameter, center, scale = make_s_curve(**data_config)
    np.savez_compressed(output / "dataset.npz", points=points, parameter=parameter, center=center, scale=scale)
    config.update(model_config=model_config, data_config=data_config, diffusion_timesteps=scheduler.num_timesteps)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if float(scheduler.alphas_cumprod[-1]) > 0.01:
        print("Warning: final alpha_bar > 0.01; the forward endpoint may retain substantial signal. "
              "Use 1000 timesteps for the experiment; short chains are for smoke tests.", flush=True)
    if not args.checkpoint:
        data = torch.from_numpy(points).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()
        log_every = max(1, min(250, args.steps // 10))
        for update in range(1, args.steps + 1):
            x = data[torch.randint(len(data), (args.batch_size,), device=device)]
            noise = torch.randn_like(x)
            t = torch.randint(scheduler.num_timesteps, (len(x),), device=device)
            noisy = scheduler.add_noise(x, noise, t)
            loss = F.mse_loss(model(noisy, t), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach().item())
            if update % log_every == 0 or update == args.steps:
                print(f"Update {update}/{args.steps} | mean MSE {np.mean(losses[-log_every:]):.5f} "
                      f"| elapsed {time.perf_counter() - start:.1f}s", flush=True)
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "model_config": model_config, "timesteps": scheduler.num_timesteps,
                    "data_config": data_config, "losses": losses, "config": config}, output / "model.pt")
    train_seconds = time.perf_counter() - start
    # Fixed independent sampling seed also reproduces trajectories after checkpoint reload.
    seed_all(args.seed + 1)
    print("Sampling reverse diffusion trajectory...", flush=True)
    trajectory, remaining = sample(model, scheduler, args.samples, args.frames, device)
    np.savez_compressed(output / "trajectory.npz", points=trajectory, remaining_steps=remaining)
    np.save(output / "loss.npy", np.asarray(losses))
    # Held-out reference uses the training normalization, with independent random draws.
    reference, _, ref_center, ref_scale = make_s_curve(args.samples, data_config["noise"], data_config["seed"] + 10000)
    reference = ((reference * ref_scale + ref_center - center) / scale).astype(np.float32)
    metrics = {"sliced_wasserstein_initial": sliced_wasserstein(trajectory[0], reference),
               "sliced_wasserstein_generated": sliced_wasserstein(trajectory[-1], reference),
               "final_alpha_bar": float(scheduler.alphas_cumprod[-1]),
               "train_or_load_seconds": train_seconds, "device": str(device),
               "torch_version": str(torch.__version__)}
    print("Rendering comparison, training curve and sampling.gif...", flush=True)
    render(output, points[:args.samples], trajectory, remaining, losses)
    metrics["total_seconds"] = time.perf_counter() - start
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
