"""生成可复用的训练数据，并从同一份数据绘制加噪结果。

在项目根目录运行：python -m lgx_ddpm.src.prepare_data
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .data import trainTarget


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "Data" / "rawdata"


def estimate_rawdata_bytes(n_samples: int, timesteps: int = 1000) -> int:
    """数组载荷：Xt、X0、epsilon 各 8 字节/点，t 为 8 字节/点。

    额外保存 float64 beta 计划；不包括 NPY 文件头、JSON 和 PNG。
    """
    return n_samples * 32 + timesteps * 8


def save_rawdata(
    output_dir: str | Path = DEFAULT_DATA_DIR,
    n_samples: int = 1_000_000,
    batch_size: int = 65536,
    beta: np.ndarray | None = None,
    noise_std: float = 0.04,
    seed: int = 42,
) -> Path:
    """分批调用 trainTarget 写入 NPY 文件；拒绝覆盖已有输出目录。

    metadata.json 仅在所有数组写入成功后生成，表示数据完整。
    可用 np.load(path, mmap_mode='r') 读取，不必一次加载全部数据。
    """
    def check_initial() -> tuple[Path, np.ndarray]:
        for name, value in (("n_samples", n_samples), ("batch_size", batch_size)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        betas = np.linspace(0.0001, 0.02, 1000) if beta is None else np.asarray(beta)
        # 复用已有参数校验；单独的随机生成器不消耗正式数据的随机序列。
        trainTarget(1, beta=betas, noise_std=noise_std, seed=seed)
        destination = Path(output_dir).resolve()
        if destination.exists():
            raise FileExistsError(f"输出目录已存在，请读取已有数据或指定新目录：{destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        required = estimate_rawdata_bytes(n_samples, len(betas)) + 1024 * 1024
        if shutil.disk_usage(destination.parent).free < required:
            raise OSError(f"磁盘空间不足，数组及元数据预计至少需要 {required} 字节")
        return destination, betas.astype(np.float64)

    destination, betas = check_initial()
    print(f"Array payload: {estimate_rawdata_bytes(n_samples, len(betas)):,} bytes", flush=True)
    destination.mkdir()
    arrays = {}
    try:
        for name in ("xt", "x0", "epsilon", "t"):
            shape = (n_samples,) if name == "t" else (n_samples, 2)
            dtype = np.int64 if name == "t" else np.float32
            arrays[name] = np.lib.format.open_memmap(
                destination / f"{name}.npy", mode="w+", dtype=dtype, shape=shape,
            )
        rng = np.random.default_rng(seed)
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            (xt, t), epsilon, x0 = trainTarget(
                stop - start, beta=betas, noise_std=noise_std, seed=rng, return_x0=True,
            )
            for name, values in (("xt", xt), ("t", t), ("epsilon", epsilon), ("x0", x0)):
                arrays[name][start:stop] = values
        for array in arrays.values():
            array.flush()
    finally:
        # Windows 上显式关闭映射，以便随后重新打开文件或移动目录。
        for array in arrays.values():
            array._mmap.close()

    np.save(destination / "beta.npy", betas, allow_pickle=False)
    metadata = {
        "format_version": 1, "status": "complete", "n_samples": n_samples,
        "batch_size": batch_size, "seed": seed, "noise_std": noise_std,
        "timesteps": len(betas), "t_indexing": "1..T (inclusive)",
        "target": "epsilon (unscaled standard Gaussian)",
        "numpy_version": np.__version__,
        "array_payload_bytes": estimate_rawdata_bytes(n_samples, len(betas)),
        "files": {"xt": "xt.npy", "t": "t.npy", "epsilon": "epsilon.npy", "x0": "x0.npy", "beta": "beta.npy"},
        "s_distribution": "u~Uniform(-3*pi/2,3*pi/2); (sin(u),sign(u)*(cos(u)-1)) + N(0,noise_std^2*I)",
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Saved {n_samples:,} samples to {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--plot-only", action="store_true", help="从已有 rawdata 重新绘图")
    args = parser.parse_args()
    if not args.plot_only:
        save_rawdata(args.output, args.samples, args.batch_size, noise_std=args.noise_std, seed=args.seed)
    # 无图形界面的环境也可保存图片；仅运行入口指定后端。
    import matplotlib
    matplotlib.use("Agg")
    from .misc.vis import vis_training_data
    for path in vis_training_data(args.output):
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
