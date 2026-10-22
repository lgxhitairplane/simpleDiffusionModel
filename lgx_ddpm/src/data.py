"""二维目标数据分布的采样函数。"""

import numpy as np


def target_S(
    n_samples: int = 10000,
    noise_std: float = 0.04,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """从 S 形目标分布 Pdata 采样，返回 (n_samples, 2) 的 float32 数组。

    分布定义：
        u ~ Uniform(-3*pi/2, 3*pi/2)
        x = sin(u)
        y = sign(u) * (cos(u) - 1)
        X0 = (x, y) + delta，delta ~ Normal(0, noise_std**2 * I_2)

    Args:
        n_samples: 采样点数，必须为正整数。
        noise_std: 目标点云厚度的标准差；为 0 时点落在曲线上。
            这是 Pdata 自身的扰动，不是扩散训练中的噪声目标。
        seed: 局部随机种子；相同种子及参数可复现相同样本。
            默认为 None，每次随机采样，不修改 NumPy 的全局随机状态。

    Returns:
        每行是一个二维点 X0。未添加厚度扰动时，x 属于 [-1, 1]，
        y 属于 [-2, 2]。函数不按当前批次归一化，以保持 Pdata 定义固定。
    """
    def check_initial() -> float:
        """校验输入参数，并返回转换为 float 的噪声标准差。"""
        if isinstance(n_samples, (bool, np.bool_)) or not isinstance(
            n_samples, (int, np.integer)
        ):
            raise TypeError("n_samples 必须为正整数")
        if n_samples <= 0:
            raise ValueError("n_samples 必须为正整数")
        if not np.isscalar(noise_std) or not np.isrealobj(noise_std):
            raise TypeError("noise_std 必须为有限的非负实数")
        checked_noise_std = float(noise_std)
        if not np.isfinite(checked_noise_std) or checked_noise_std < 0:
            raise ValueError("noise_std 必须为有限的非负实数")
        return checked_noise_std

    noise_std = check_initial()

    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.5 * np.pi, 1.5 * np.pi, size=n_samples)
    points = np.column_stack((np.sin(u), np.sign(u) * (np.cos(u) - 1)))
    if noise_std > 0:
        points += rng.normal(0.0, noise_std, size=points.shape)
    return points.astype(np.float32)


# 新名称与已有 target_S 使用同一套 S 形采样逻辑。
sample_target_s = target_S


def trainTarget(
    n_samples: int = 256,
    beta: np.ndarray | None = None,
    noise_std: float = 0.04,
    seed: int | np.random.Generator | None = None,
    *,
    return_x0: bool = False,
) -> (
    tuple[tuple[np.ndarray, np.ndarray], np.ndarray]
    | tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]
):
    """返回一批 DDPM 训练输入 ((Xt, t), epsilon)。

    beta 是完整的一维方差计划；默认 1000 步，从 0.0001 线性增加到 0.02。
    noise_std 只控制 S 形分布厚度，epsilon 始终服从二维标准高斯分布。
    Xt、epsilon 为 (n_samples, 2) 的 float32 数组。
    t 为 (n_samples,) 的 int64 数组，范围为 1..T（含两端）。
    seed 支持整数或 NumPy Generator；连续训练请复用 Generator 或传 None，
    反复传同一个整数会重复生成相同批次。
    return_x0=True 时额外返回对应的原始点：((Xt, t), epsilon, X0)。
    """
    def check_initial() -> np.ndarray:
        """校验 beta；样本数和厚度由 S 形采样函数校验。"""
        if beta is None:
            return np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
        values = np.asarray(beta)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("beta 必须为非空的一维方差计划")
        if not np.issubdtype(values.dtype, np.number) or not np.isrealobj(values):
            raise TypeError("beta 必须包含实数")
        values = values.astype(np.float64)
        if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
            raise ValueError("beta 中每个值必须满足 0 < beta_t < 1")
        return values

    betas = check_initial()
    rng = np.random.default_rng(seed)

    # 独立采样 X0、t、epsilon；传递同一个生成器以避免重复设置种子。
    x0 = sample_target_s(n_samples=n_samples, noise_std=noise_std, seed=rng)
    t = rng.integers(1, len(betas) + 1, size=n_samples, dtype=np.int64)
    epsilon = rng.standard_normal(size=(n_samples, 2)).astype(np.float32)

    alpha_bar = np.cumprod(1.0 - betas)
    # t 从 1 开始，数组索引从 0 开始；增加一维以广播到 x、y 两个坐标。
    alpha_bar_t = alpha_bar[t - 1, None]
    xt = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * epsilon
    inputs = (xt.astype(np.float32), t)
    if return_x0:
        return inputs, epsilon, x0
    return inputs, epsilon
