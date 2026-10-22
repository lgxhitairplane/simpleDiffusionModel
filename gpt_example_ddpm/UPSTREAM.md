# 上游来源记录

本实验复用了 [tanelp/tiny-diffusion](https://github.com/tanelp/tiny-diffusion) 的二维 DDPM 实现。

参考分支：`master`；参考日期：2026-10-22。以下 SHA256 对应本地用于改造的原始文件，可与 `temp/upstream/` 中的副本校验：

| 原文件 | SHA256 |
|---|---|
| `ddpm.py` | `FAA1399945F426145A6E1AA3CBA5FE5483B922A7D83023FC02458183F5A1A38C` |
| `positional_embeddings.py` | `9F68C6733E317040C655EC49E3273146EAC8B2FD3DB9B4664CA9DD2C9C96E764` |

`src/tiny_diffusion.py` 保留了上游 `Block`、`MLP`、`NoiseScheduler` 的结构与计算公式，移除原训练入口，增加设备 buffer 管理与参数校验。`src/positional_embeddings.py` 保留上游编码实现，将正弦频率缓存到设备 buffer，并校验编码大小。

S 形数据、训练入口、结果保存、可视化和测试由本实验补充。上游文件的哈希标记原始内容，不等同于 Git commit 标识。
