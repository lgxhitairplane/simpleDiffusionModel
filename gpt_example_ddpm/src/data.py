"""Seeded 2D S-curve dataset with finite thickness."""
import numpy as np


def make_s_curve(count=20000, noise=0.04, seed=42):
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1.5 * np.pi, 1.5 * np.pi, count)
    points = np.column_stack((np.sin(t), np.sign(t) * (np.cos(t) - 1)))
    points += rng.normal(0, noise, points.shape)
    # One common scale preserves the shape's aspect ratio.
    center = points.mean(axis=0)
    scale = np.sqrt(np.mean((points - center) ** 2))
    return ((points - center) / scale).astype(np.float32), t.astype(np.float32), center, scale


def sliced_wasserstein(a, b, seed=123, projections=128):
    """Mean 1D W1 over random projections; smaller is better, not a pass/fail score."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(2, projections))
    directions /= np.linalg.norm(directions, axis=0)
    q = (np.arange(min(len(a), len(b))) + 0.5) / min(len(a), len(b))
    return float(np.abs(np.quantile(a @ directions, q, axis=0)
                        - np.quantile(b @ directions, q, axis=0)).mean())
