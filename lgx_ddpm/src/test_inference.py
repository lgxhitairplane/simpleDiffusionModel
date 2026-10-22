"""核验 DDPM 更新公式、时间编号、随机数及模型状态。"""

import unittest

import numpy as np
import torch
from torch import nn

from .model import inference


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(self, x, t):
        self.calls.append((t.clone(), self.training, torch.is_grad_enabled()))
        return self.scale * x


class InferenceTests(unittest.TestCase):
    def test_recording_keeps_rng_and_captures_real_states(self):
        model = RecordingModel()
        expected = inference(model, [0.1, 0.2, 0.3], 4, seed=7)
        frames = {}
        result, initial = inference(model, [0.1, 0.2, 0.3], 4, seed=7,
                                    return_initial=True, frame_every=2,
                                    on_frame=lambda t, x: frames.update({t: x}))
        self.assertEqual(list(frames), [3, 1, 0])
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(frames[3], initial)
        torch.testing.assert_close(frames[0], result)
        # 回调收到独立副本，修改帧不影响最终结果。
        frames[0].zero_()
        torch.testing.assert_close(result, expected)

    def test_two_steps_match_formula_and_preserve_state(self):
        model = RecordingModel()
        global_state = torch.random.get_rng_state().clone()
        actual, initial = inference(model, np.array([0.1, 0.2]), 4, seed=7, return_initial=True)
        rng = torch.Generator().manual_seed(7)
        start = torch.randn((4, 2), generator=rng)
        z = torch.randn((4, 2), generator=rng)
        # t=2: alpha_bar=.9*.8=.72; posterior variance=.2*.1/.28。
        middle = (start - 0.2 / np.sqrt(0.28) * 0.25 * start) / np.sqrt(0.8)
        middle += np.sqrt(0.2 * 0.1 / 0.28) * z
        # t=1: 仅均值，没有第三次高斯抽样。
        expected = (middle - 0.1 / np.sqrt(0.1) * 0.25 * middle) / np.sqrt(0.9)
        torch.testing.assert_close(initial, start)
        torch.testing.assert_close(actual, expected)
        self.assertEqual([call[0].tolist() for call in model.calls], [[1]*4, [0]*4])
        self.assertTrue(all(not training and not grad for _, training, grad in model.calls))
        self.assertTrue(model.training)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(torch.random.get_rng_state(), global_state)
        torch.testing.assert_close(inference(model, [0.1, 0.2], 4, seed=7), actual)

    def test_one_step_and_invalid_schedule(self):
        model = RecordingModel().eval()
        result, initial = inference(model, [0.36], 3, return_initial=True)
        torch.testing.assert_close(result, (initial - 0.6 * 0.25 * initial) / 0.8)
        self.assertFalse(model.training)
        for beta in ([], [0], [1], [float("nan")], [[0.1]]):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                inference(model, beta, 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_device_and_reproducibility(self):
        model = RecordingModel().cuda().eval()
        first = inference(model, [0.1, 0.2], 4, seed=8)
        second = inference(model, [0.1, 0.2], 4, seed=8)
        self.assertEqual(first.device.type, "cuda")
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
