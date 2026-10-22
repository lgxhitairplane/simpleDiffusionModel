"""Numerical and device checks; run with unittest discovery."""
import unittest
import numpy as np
import torch
from data import make_s_curve, sliced_wasserstein
from tiny_diffusion import MLP, NoiseScheduler
from main import sample, seed_all


class DiffusionTests(unittest.TestCase):
    def test_data_reproducibility(self):
        a = make_s_curve(1024, seed=2)[0]
        b = make_s_curve(1024, seed=2)[0]
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (1024, 2))
        self.assertTrue(np.isfinite(a).all())
        self.assertAlmostEqual(float(np.mean(a ** 2)), 1.0, places=5)

    def test_forward_noise_can_be_inverted(self):
        scheduler = NoiseScheduler()
        x, noise = torch.randn(4, 2), torch.randn(4, 2)
        t = torch.tensor([0, 10, 500, 999])
        recovered = scheduler.reconstruct_x0(scheduler.add_noise(x, noise, t), t, noise)
        torch.testing.assert_close(recovered, x, atol=5e-5, rtol=5e-5)
        self.assertLess(scheduler.alphas_cumprod[-1].item(), 0.001)

    def test_last_step_adds_no_noise(self):
        scheduler = NoiseScheduler()
        x, noise = torch.randn(8, 2), torch.randn(8, 2)
        noisy = scheduler.add_noise(x, noise, torch.zeros(8, dtype=torch.long))
        first = scheduler.step(noise, 0, noisy)
        second = scheduler.step(noise, 0, noisy)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        # Float32 posterior coefficients have cancellation error near t=0.
        torch.testing.assert_close(first, x, atol=1e-3, rtol=1e-3)

    def test_metric_identity_and_shift(self):
        x = make_s_curve(200)[0]
        self.assertAlmostEqual(sliced_wasserstein(x, x), 0)
        self.assertGreater(sliced_wasserstein(x, x + 2), 0.5)

    def check_device(self, device):
        model = MLP(hidden_size=32, hidden_layers=1, emb_size=16).to(device)
        scheduler = NoiseScheduler(10).to(device)
        self.assertTrue(all(b.device.type == device for b in scheduler.buffers()))
        x = torch.randn(32, 2, device=device)
        t = torch.randint(10, (32,), device=device)
        loss = (model(scheduler.add_noise(x, torch.randn_like(x), t), t) ** 2).mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        seed_all(8)
        first, steps = sample(model, scheduler, 32, 6, device)
        other = MLP(hidden_size=32, hidden_layers=1, emb_size=16).to(device)
        other.load_state_dict(model.state_dict())
        seed_all(8)
        second, _ = sample(other, scheduler, 32, 6, device)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (6, 32, 2))
        self.assertEqual(steps[0], 10)
        self.assertEqual(steps[-1], 0)

    def test_cpu(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        self.check_device("cuda")


if __name__ == "__main__":
    unittest.main()
