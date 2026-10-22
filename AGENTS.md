# Diffusion Model 实验约定

本文档供参与本项目的开发者和大模型查阅。后续开发应遵循以下目标与目录约定。

## 实验目标

- 训练一个 Diffusion Model（扩散模型），在二维（2D）空间中实现从高斯分布到 S 形分布的生成映射。
- 可视化采样生成的动态过程，展示样本如何从高斯噪声逐步演化为 S 形分布。

## 目录约定

```text
diffusion/
├── AGENTS.md   # 项目目标与开发约定
├── Data/       # 数据
├── temp/       # 临时文件及临时使用的子目录
└── src/        # 主要源码
```

- `Data/`：统一存放实验数据；目录名保留大写 `D`。
- `temp/`：统一存放仅供临时使用的文件和文件夹。
- `src/`：统一存放主要源码，包括后续的数据处理、模型、训练、采样及可视化代码。
- 后续新增或修改项目约定时，同步更新本文档，方便其他开发者和大模型继续工作。

## 首个可运行实验

- 用户已选定基于 `tanelp/tiny-diffusion` 实现首版实验，位于 `gpt_example_ddpm/`。
- 该实验内独立设置 `Data/`、`temp/`、`src/`。持久实验数据、模型及图表统一保存在 `gpt_example_ddpm/Data/runs/`；临时缓存和上游参考文件放在该实验的 `temp/`。
- 使用 PyTorch，支持 CUDA GPU 训练与采样；网络沿用 tiny-diffusion 的二维残差 MLP 和时间编码。
- 首版默认采用参数化 S 形点云、小幅高斯扰动及保持纵横比例的归一化；使用 DDPM 线性噪声调度、1,000 个扩散步、5,000 次训练更新。
- 可视化包括真实逐步去噪轨迹 GIF、目标/初始/最终分布对比及损失曲线。
- 参数定义、运行方法和来源详见 `gpt_example_ddpm/README.md`。这些为首版默认实现，后续实验可通过命令行调整，并保存配置。

## 用户的逐步实现思路（2026-10-22）

本节记录用户提出的高斯分布 → S 形分布实现思路。当前先记录，后续按用户的节奏逐项展开和实现；现有示例的默认配置不代表本轮已经确定的方案。

### 0. 定义模型输入、输出及 loss

- 模型输入：带噪二维点 `Xt` 和时间步 `t`。
- 模型输出：噪声预测，即 `model(Xt, t) = noise` 为期望学习的关系。
- 损失函数：`Loss = mse(noise, model(Xt, t))`；其中 `noise` 为训练目标，`model(Xt, t)` 为模型预测。

### 1. 制造数据，即制造 train、target

1. 定义 S 形分布 `Pdata`。
2. 定义 `trainTarget` 的获取方式，即如何构造训练输入与对应目标。

上述两项已在 `lgx_ddpm/src/data.py` 实现，按下述定义采样目标点和训练样本。

### 当前 S 形分布定义

- 逐步实现代码位于 `lgx_ddpm/src/`，数据及临时文件分别使用该实验内的 `Data/` 和 `temp/`。
- `sample_target_s(n_samples=10000, noise_std=0.04, seed=None)` 从目标分布采样，返回形状为 `(n_samples, 2)` 的 NumPy `float32` 数组，每行是一个原始二维点 `X0`。保留原函数 `target_S`，并提供 `sample_target_s = target_S` 别名。
- 令 `u ~ Uniform(-3π/2, 3π/2)`，曲线坐标为 `(sin(u), sign(u) * (cos(u) - 1))`；再添加标准差为 `noise_std` 的各向同性高斯扰动，形成有厚度的 S 形点云。
- `noise_std=0` 表示无厚度曲线；这里的扰动属于 `Pdata` 本身，不是后续扩散训练的噪声目标。
- 使用局部随机种子控制复现，不按当前采样批次归一化，保持目标分布定义固定。
- `lgx_ddpm/src/misc/vis.py` 中的 `vis_s_distribution(s_distribution, *, save_path=None, show=True)` 接收 `(N, 2)` 样本绘制等比例散点图，支持显示及保存图片，返回 Matplotlib 的 `(fig, ax)`。持久图片放在 `lgx_ddpm/Data/`，临时检查图片和绘图库缓存放在 `lgx_ddpm/temp/`。

### 当前训练样本构造

- `trainTarget(n_samples=256, beta=None, noise_std=0.04, seed=None)` 调用 `sample_target_s` 抽取 `X0`，独立均匀抽取整数时间步 `t` 和二维标准高斯噪声 `epsilon`。
- `beta` 为完整的一维方差计划，各项满足 `0 < beta_t < 1`；默认采用原论文的 1000 步线性计划，从 `0.0001` 增加至 `0.02`。
- `alpha_t = 1 - beta_t`，`alpha_bar_t` 是从第 1 步至第 t 步的 alpha 累乘；`Xt = sqrt(alpha_bar_t) * X0 + sqrt(1-alpha_bar_t) * epsilon`。
- 返回 `((Xt, t), epsilon)`；`Xt`、`epsilon` 形状为 `(N, 2)`、类型为 `float32`，`t` 形状为 `(N,)`、类型为 `int64`，取值为 `1..T`，查数组时使用 `t-1`。
- 采样函数支持整数种子或 NumPy Generator；连续训练应复用 Generator 或使用 `seed=None`，避免每次重置为同一整数种子而重复同一批数据。
- `trainTarget(..., return_x0=True)` 额外返回原始点，格式为 `((Xt, t), epsilon, X0)`；默认返回格式不变。

### 训练数据保存与加噪可视化

- 项目根目录运行 `python -m lgx_ddpm.src.prepare_data`，默认分批生成 100 万条样本至 `lgx_ddpm/Data/rawdata/`，已有目录不会被覆盖；`--plot-only` 可只读取已有数据重新绘图。
- 保存 `xt.npy`、`t.npy`、`epsilon.npy`、`x0.npy` 和 `beta.npy`；`metadata.json` 记录参数、随机种子、批次大小、NumPy 版本及完整状态。训练用 `np.load(..., mmap_mode='r')` 按需读取 `Xt,t,epsilon`，`X0` 保留用于核对和可视化。
- 每条样本占 32 字节（3 个二维 float32 向量 + 1 个 int64 时间步）；100 万条样本的数组载荷约 32 MB / 30.5 MiB，另加 beta、文件头、元数据和 PNG。生成前估算并检查磁盘可用空间。
- `lgx_ddpm/src/misc/vis.py` 的 `group_training_data` 分块读取并按精确时间步汇总；`vis_training_data` 为 `t=100,300,500,700,1000` 各保存一张 `Xt / X0 / epsilon` 三联图至 `rawdata/plots/`。同组样本严格对应，跨组坐标范围相同；epsilon 是未缩放的训练目标。
- 时间步仍按均匀分布随机抽样，100 万条样本中每个 t 预计约 1000 条，实际数量随机；不同 t 的图展示不同原始样本的分布，不表示相同样本的演化轨迹。

## 当前模型训练

- `lgx_ddpm/src/model.py` 复用 `gpt_example_ddpm/src/tiny_diffusion.py` 的默认 MLP（128 维坐标/时间编码、128 隐藏宽度、3 个残差块，99,074 个参数），随机初始化后训练。
- `Loss(model, Xt, t_model, epsilon)` 直接计算模型预测与目标噪声的均方误差，对 batch 和坐标取平均；`t_model` 为 `0..T-1`。训练读取保存的 `t` 后减 1，后续采样必须沿用这个编号约定。
- `trainModel` 默认从 `lgx_ddpm/Data/rawdata/` 按需读取，训练 5000 步、batch size 512、Adam 学习率 0.0002、梯度最大范数 1；固定随机划分 10000 条验证样本，每 500 步及末步评估，训练样本按批次有放回抽取。数据保持原尺度。
- 默认自动选择 CUDA/CPU；输出至 `lgx_ddpm/Data/runs/tiny_<时间戳>/`，拒绝覆盖。保存 `best.pt`、`last.pt`、`config.json`、`validation_indices.npy`、`train_loss.npy`、`history.json`、`summary.json`；检查点包含结构配置、beta 计划和时间步约定。返回验证误差最低的模型（eval 模式），暂不支持断点续训。
- 从项目根目录运行 `.\gpt_example_ddpm\.venv\Scripts\python.exe -m lgx_ddpm.src.model --steps 5000`；短程验证可降低 `--steps`、`--validation-size` 并调整 `--eval-every`。模块依赖 NumPy 和 PyTorch，复用已安装的实验环境。
- 模型检查：`.\gpt_example_ddpm\.venv\Scripts\python.exe -m unittest lgx_ddpm.src.test_model -v`。

## 版本管理

- 使用 Git 管理项目，GitHub 远程仓库为 `https://github.com/lgxhitairplane/simpleDiffusionModel`，主分支为 `main`。
- 使用 Git 提交记录可恢复的版本，不再通过复制源码到 temp 目录维护核验副本。
- 源码、项目约定和依赖清单纳入版本管理；临时缓存、虚拟环境由根目录 `.gitignore` 排除，temp 目录占位文件 `.gitkeep` 可保留；所有 `Data/` 目录整体忽略（包括其占位文件）。
- 提交前检查差异及暂存清单，确认必要验证通过。Git 提交保存本地版本；推送至远程后才会同步到 GitHub。
