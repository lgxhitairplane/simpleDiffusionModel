"""Loss、训练保存和独立验证集重载检查。"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from .model import Loss, MLP, trainModel
from .prepare_data import save_rawdata


class ModelTests(unittest.TestCase):
    def test_loss_value_and_gradient(self):
        class ScalingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(2.0))

            def forward(self, x, t):
                return self.scale * x

        model = ScalingModel()
        loss = Loss(model, torch.tensor([[1.0, 2.0]]), torch.tensor([0]), torch.zeros(1, 2))
        self.assertEqual(loss.item(), 10.0)
        loss.backward()
        self.assertEqual(model.scale.grad.item(), 10.0)

    def test_loss_rejects_target_broadcasting(self):
        with self.assertRaises(ValueError):
            Loss(nn.Identity(), torch.ones(3, 2), torch.zeros(3, dtype=torch.long), torch.zeros(3, 1))

    def test_training_checkpoint_and_validation(self):
        temp_root = Path(__file__).resolve().parents[1] / "temp"
        temp_root.mkdir(exist_ok=True)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
                root = Path(tmp)
                raw = save_rawdata(root / "raw", 65, batch_size=17, beta=[.05, .2, .4], seed=7)
                output = root / "run"
                model = trainModel(raw, output_dir=output, steps=2, batch_size=8,
                                   validation_size=17, eval_every=1, device="cpu", seed=12)
                checkpoint = torch.load(output / "best.pt", weights_only=True)
                restored = MLP(**checkpoint["config"]["architecture"])
                restored.load_state_dict(checkpoint["model_state_dict"])
                restored.eval()
                rows = np.load(output / "validation_indices.npy")
                self.assertEqual(len(np.unique(rows)), 17)
                xt = torch.from_numpy(np.load(raw / "xt.npy")[rows])
                t = torch.from_numpy(np.load(raw / "t.npy")[rows] - 1)
                epsilon = torch.from_numpy(np.load(raw / "epsilon.npy")[rows])
                with torch.no_grad():
                    torch.testing.assert_close(model(xt, t), restored(xt, t))
                    measured = Loss(restored, xt, t, epsilon).item()
                self.assertAlmostEqual(measured, checkpoint["validation_loss"], places=5)
                np.testing.assert_allclose(checkpoint["beta"].numpy(), [.05, .2, .4])
                last = torch.load(output / "last.pt", weights_only=True)
                self.assertEqual(last["step"], 2)
                self.assertEqual(np.load(output / "train_loss.npy").shape, (2,))
                self.assertEqual((output / "loss.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertFalse(model.training)
                with self.assertRaises(FileExistsError):
                    trainModel(raw, output_dir=output, validation_size=17, device="cpu")
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
