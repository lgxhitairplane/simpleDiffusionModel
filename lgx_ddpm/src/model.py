"""复用 tiny-diffusion MLP，在保存的二维 DDPM 样本上训练。

从项目根目录运行：python -m lgx_ddpm.src.model --steps 5000
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gpt_example_ddpm.src.tiny_diffusion import MLP


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"


def Loss(
    model: nn.Module,
    Xt: torch.Tensor,
    t_model: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """DDPM 简化目标：MSE(model(Xt, t_model), epsilon)。

    Xt、epsilon 为 (B, 2)，t_model 为 (B,) 且使用 0..T-1 编号。
    对 batch 和两个坐标取平均；保留计算图供 loss.backward() 使用。
    """
    if Xt.ndim != 2 or Xt.shape[1] != 2 or epsilon.shape != Xt.shape:
        raise ValueError("Xt 和 epsilon 必须具有相同的 (B, 2) 形状")
    if t_model.shape != (Xt.shape[0],):
        raise ValueError("t_model 必须具有 (B,) 形状")
    prediction = model(Xt, t_model)
    if prediction.shape != epsilon.shape:
        raise ValueError("模型输出形状必须与 epsilon 相同")
    return F.mse_loss(prediction, epsilon)


def trainModel(
    rawdata_dir: str | Path = DATA_DIR / "rawdata",
    *,
    output_dir: str | Path | None = None,
    steps: int = 5000,
    batch_size: int = 512,
    learning_rate: float = 2e-4,
    validation_size: int = 10000,
    eval_every: int = 500,
    seed: int = 42,
    device: str = "auto",
) -> nn.Module:
    """训练 tiny-diffusion 默认 MLP，返回验证 loss 最低的模型（eval 模式）。

    每步从训练集有放回随机抽取 batch_size 条数据，执行一次参数更新。
    固定随机划分 validation_size 条验证数据，验证集不参与梯度更新。
    数据按需读取，只有当前 batch 传入设备；不重新归一化已加噪的数据。
    默认训练 5000 步，使用 Adam 和最大范数为 1 的梯度裁剪。

    输出目录默认为 Data/runs/tiny_<时间戳>，拒绝覆盖已有目录。
    best.pt、last.pt 保存权重、架构、beta 和时间步约定等推理所需信息。
    train_loss.npy 保存逐步 loss，history.json 保存验证记录，loss.png 保存曲线。
    本函数每次从头训练，暂不提供断点续训接口。
    """
    def check_initial():
        for name, value in (("steps", steps), ("batch_size", batch_size),
                            ("validation_size", validation_size), ("eval_every", eval_every)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate 必须为有限正数")
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA")
        root = Path(rawdata_dir).resolve()
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "complete" or metadata.get("t_indexing") != "1..T (inclusive)":
            raise ValueError("需要完整的 rawdata，且时间步约定为 1..T")
        if not 0 < validation_size < metadata["n_samples"]:
            raise ValueError("validation_size 必须小于数据集样本数")
        beta = np.load(root / "beta.npy", allow_pickle=False)
        if beta.shape != (metadata["timesteps"],) or not np.isfinite(beta).all() or np.any((beta <= 0) | (beta >= 1)):
            raise ValueError("beta 计划与元数据不一致或包含无效值")
        destination = Path(output_dir).resolve() if output_dir is not None else DATA_DIR / "runs" / datetime.now().strftime("tiny_%Y%m%d_%H%M%S_%f")
        if destination.exists():
            raise FileExistsError(f"训练输出目录已存在：{destination}")
        return root, metadata, beta, selected, destination

    root, metadata, beta, selected, destination = check_initial()
    arrays = {}
    try:
        for name in ("xt", "t", "epsilon"):
            arrays[name] = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            shape = (metadata["n_samples"],) if name == "t" else (metadata["n_samples"], 2)
            dtype = np.int64 if name == "t" else np.float32
            if arrays[name].shape != shape or arrays[name].dtype != dtype:
                raise ValueError(f"{name}.npy 形状或类型不符合 rawdata 约定")
        for start in range(0, metadata["n_samples"], 65536):
            end = start + 65536
            if not all(np.isfinite(a[start:end]).all() for a in arrays.values()):
                raise ValueError("训练数据包含 NaN 或无穷大")
            if np.any((arrays["t"][start:end] < 1) | (arrays["t"][start:end] > len(beta))):
                raise ValueError("训练数据中的 t 超出 1..T")

        torch.manual_seed(seed)
        if selected.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(metadata["n_samples"])
        val_indices, train_indices = indices[:validation_size], indices[validation_size:]
        architecture = dict(hidden_size=128, hidden_layers=3, emb_size=128,
                            time_emb="sinusoidal", input_emb="sinusoidal")
        model = MLP(**architecture).to(selected)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def get_batch(rows):
            xt = torch.from_numpy(arrays["xt"][rows]).to(selected)
            # 仅在入口转换一次：保存的 t 为 1..T，模型使用 0..T-1。
            t_model = torch.from_numpy(arrays["t"][rows] - 1).to(selected)
            epsilon = torch.from_numpy(arrays["epsilon"][rows]).to(selected)
            return xt, t_model, epsilon

        def evaluate():
            model.eval()
            total = 0.0
            with torch.no_grad():
                for start in range(0, validation_size, batch_size):
                    rows = val_indices[start:start + batch_size]
                    total += Loss(model, *get_batch(rows)).item() * len(rows)
            result = total / validation_size
            if not np.isfinite(result):
                raise RuntimeError("验证 loss 非有限，训练停止")
            return result

        config = dict(architecture=architecture, steps=steps, batch_size=batch_size,
                      learning_rate=learning_rate, validation_size=validation_size,
                      eval_every=eval_every, seed=seed, device=str(selected),
                      rawdata_dir=str(root), data_metadata=metadata,
                      model_t_indexing="0..T-1; model_t = raw_t - 1", normalization="none",
                      optimizer="Adam", gradient_clip_norm=1.0,
                      torch_version=str(torch.__version__), numpy_version=np.__version__,
                      parameter_count=sum(p.numel() for p in model.parameters()))
        destination.mkdir(parents=True)
        (destination / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        # 明确保存验证行号，便于检查划分或独立复核验证误差。
        np.save(destination / "validation_indices.npy", val_indices, allow_pickle=False)

        def save_checkpoint(name, step, validation_loss):
            torch.save({
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config, "beta": torch.from_numpy(beta.copy()),
                "step": step, "validation_loss": validation_loss,
            }, destination / name)

        losses = []
        initial_loss = evaluate()
        best_loss, best_step = initial_loss, 0
        history = [{"step": 0, "train_loss": None, "validation_loss": initial_loss}]
        save_checkpoint("best.pt", 0, initial_loss)
        started = time.perf_counter()
        print(f"Training on {selected}; parameters={config['parameter_count']:,}; output={destination}", flush=True)
        print(f"Initial validation MSE: {initial_loss:.6f}", flush=True)
        for step in range(1, steps + 1):
            model.train()
            rows = train_indices[rng.integers(0, len(train_indices), size=batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = Loss(model, *get_batch(rows))
            if not torch.isfinite(loss):
                raise RuntimeError(f"第 {step} 步 loss 非有限，训练停止")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            if step % eval_every == 0 or step == steps:
                validation_loss = evaluate()
                mean_loss = float(np.mean(losses[history[-1]["step"]:step]))
                history.append({"step": step, "train_loss": mean_loss, "validation_loss": validation_loss})
                if validation_loss < best_loss:
                    best_loss, best_step = validation_loss, step
                    save_checkpoint("best.pt", step, validation_loss)
                save_checkpoint("last.pt", step, validation_loss)
                np.save(destination / "train_loss.npy", np.asarray(losses, dtype=np.float32), allow_pickle=False)
                (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                print(f"step {step}/{steps}: train={mean_loss:.6f}, validation={validation_loss:.6f}", flush=True)

        summary = dict(best_step=best_step, best_validation_loss=best_loss,
                       initial_validation_loss=initial_loss, final_validation_loss=history[-1]["validation_loss"],
                       elapsed_seconds=time.perf_counter() - started)
        (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        from .misc.vis import vis_loss
        fig, _ = vis_loss(destination)
        fig.clear()
        print(f"Saved loss plot: {destination / 'loss.png'}", flush=True)
        checkpoint = torch.load(destination / "best.pt", map_location=selected, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    finally:
        for array in arrays.values():
            array._mmap.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawdata", type=Path, default=DATA_DIR / "rawdata")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    trainModel(args.rawdata, output_dir=args.output, steps=args.steps,
               batch_size=args.batch_size, learning_rate=args.learning_rate,
               validation_size=args.validation_size, eval_every=args.eval_every,
               seed=args.seed, device=args.device)
