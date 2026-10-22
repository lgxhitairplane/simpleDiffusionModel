# 二维 S 形分布 DDPM 实验

基于 [tanelp/tiny-diffusion](https://github.com/tanelp/tiny-diffusion) 的二维 MLP 与 DDPM 实现，训练从二维标准高斯噪声生成 S 形点分布的模型，并导出一次完整反向采样过程的动画。支持 CUDA GPU 和 CPU。

## 运行

本机环境位于 `.venv`。在本文件所在目录打开 PowerShell：

```powershell
.\.venv\Scripts\python.exe src/main.py --device cuda
```

默认训练 5,000 次参数更新，20,000 个训练点，batch size 512，1,000 个扩散时间步，生成 2,000 个点并保存 101 帧动画。`--steps` 是训练更新次数，`--timesteps` 是每次生成的去噪步数，两者含义不同。

不指定 `--device` 时自动选择 CUDA，无法使用 CUDA 时选择 CPU。显式指定 `--device cuda` 时，如 GPU 不可用会报错。所有数据、模型及训练张量均在选定设备上计算；图像和 GIF 使用 CPU 绘制。

自定义实验：

```powershell
.\.venv\Scripts\python.exe src/main.py --device cuda --steps 10000 --samples 3000 --run-name longer_training
```

运行名称不可重复，以免覆盖已有实验。默认以时间生成新目录。路径以项目目录为基准，不依赖命令执行时的工作目录。

## 新环境安装

建议 Python 3.11–3.13。CUDA 12.8 版 PyTorch 适用于本机 RTX 5070；其他设备请参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。已有合适的 GPU PyTorch 环境也可直接使用。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU 环境可把 PyTorch 下载地址改成 `https://download.pytorch.org/whl/cpu`，运行时使用 `--device cpu`。

## 数据与模型

- 参数 `u` 均匀采样于 `[-1.5π, 1.5π]`，二维坐标为 `(sin(u), sign(u) × (cos(u) − 1))`。
- 给两个坐标加入标准差为 0.04 的高斯噪声，形成有厚度的 S 形带；随后减去训练数据均值，使用共同的尺度归一化，保持纵横比例。可用 `--noise` 修改厚度。
- 使用上游残差 MLP，分别编码两个坐标和时间步。默认编码维度 128、隐藏宽度 128、3 个残差块，输出二维噪声预测。
- 使用原始 DDPM 噪声预测 MSE 目标、AdamW、梯度裁剪，线性 beta 从 0.0001 增至 0.02。
- 默认 1,000 步时末端 alpha_bar 约为 0.000040，接近纯噪声。不要只把扩散步数降到几十步而保持相同 beta 范围来评估生成质量；短链只适合快速检查程序。
- 不需要预训练权重：复用上游算法和网络结构，在本项目合成数据上训练。

## 输出

每次运行保存在 `Data/runs/<运行名称>/`：

| 文件 | 内容 |
|---|---|
| `dataset.npz` | 归一化后的训练点、曲线参数、中心和尺度 |
| `config.json` | 本次参数及实际使用设备 |
| `model.pt` | 网络权重、结构、数据参数和训练损失，训练模式生成 |
| `trajectory.npz` | `[帧数, 点数, 2]` 坐标与每帧剩余去噪步数 |
| `sampling.gif` | 目标分布与逐步反向采样的并排动画 |
| `comparison.png` | 目标 S 形、初始高斯噪声、最终样本 |
| `loss.png` / `loss.npy` | 训练损失曲线及数值 |
| `metrics.json` | 初始/生成分布到独立参考数据的切片 Wasserstein 距离和运行时间 |

动画记录同一批点沿真实 DDPM 随机采样链移动；颜色和点编号保持一致，并展示少量短轨迹。没有对起点与终点做插值。保存帧会跳过部分中间时间步，但采样仍执行全部去噪步。坐标轴在动画中固定。

指标越小表示分布越接近，只作诊断；应同时观察 S 形覆盖范围、厚度和异常点。验证参考数据使用独立随机种子，并沿用训练数据的归一化参数。

## 读取已有模型重新采样

```powershell
.\.venv\Scripts\python.exe src/main.py --device cuda --checkpoint Data/runs/<运行名称>/model.pt --run-name resample
```

此模式跳过训练，使用 checkpoint 保存的网络、扩散步数和数据设置。使用相同采样种子、样本数和设备，可重现采样轨迹；不同硬件和软件版本之间不承诺逐位一致。`model.pt` 用于推理重载，当前不提供继续训练的优化器恢复功能。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_core.py -v
```

覆盖可重复数据生成、加噪逆变换、最后一步不再添加随机噪声、分布距离、CPU/CUDA 前向反向计算及权重重载后的采样一致性。

## 源码与来源

- `src/tiny_diffusion.py`：来自上游 `ddpm.py` 的 MLP、残差块和 DDPM 调度器；增加调度系数的 buffer 注册，使其随设备迁移。
- `src/positional_embeddings.py`：来自上游同名文件；把正弦编码频率改成注册 buffer，修复 GPU 设备不匹配。
- `src/data.py`：S 形数据及分布距离。
- `src/main.py`：训练、检查点与采样入口。
- `src/visualize.py`：静态图和动画。

上游来源为 `https://github.com/tanelp/tiny-diffusion` 的 master 分支，本次参考日期为 2026-10-22；参考原文件保存在 `temp/upstream/`。运行代码不依赖该临时目录。

## 本机实测

2026-10-22 在 NVIDIA GeForce RTX 5070 Laptop GPU 上完成默认 5,000 次训练更新，环境为 Python 3.13.9、PyTorch 2.7.1+cu128、NumPy 2.3.5、Matplotlib 3.10.6、Pillow 12.0.0。结果目录为 `Data/runs/first_gpu/`。

- 训练耗时约 71 秒（包含训练初始化，不含 Python 启动、采样与图像导出）。
- 2,000 个生成点，保存 101 帧真实反向采样坐标。
- 对独立参考数据的切片 Wasserstein 距离，从初始噪声约 0.2652 降至生成样本约 0.0539；生成图已形成 S 形，仍有少量偏离曲线的点。
- 6 项单元检查通过，包含 CPU 与 CUDA。
- `Data/runs/reload_check/` 已验证从磁盘加载模型后独立完成 GPU 采样。
