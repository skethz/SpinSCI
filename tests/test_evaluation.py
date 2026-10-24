import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tools import evaluate


ROOT = Path(__file__).resolve().parents[1]


class EvaluationPrecision(unittest.TestCase):
    def test_gray_measurements_preserve_source_precision(self):
        rng = np.random.default_rng(109)
        source = rng.uniform(0, 255, (16, 18, 8))
        mask = rng.integers(0, 2, source.shape).astype(np.float32)
        expected = ((source / 255) * mask).sum(2).astype(np.float32)
        rounded_early = ((source.astype(np.float32) / 255) * mask).sum(2)
        self.assertFalse(np.array_equal(expected, rounded_early))
        measurements = []
        mask_strides = []

        class CaptureMeasurement(torch.nn.Module):
            def forward(self, y, phi, phi_s):
                measurements.append(y.detach().cpu().numpy().copy())
                mask_strides.append((phi.stride()[-2:], phi_s.stride()[-2:]))
                return [y * phi / phi_s]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "sequence.npy", source)
            np.save(root / "mask.npy", mask)
            arguments = ["evaluate", str(ROOT / "configs/gray_cr8.json"),
                         "--checkpoint", "unused.pth", "--input", str(root / "sequence.npy"),
                         "--mask", str(root / "mask.npy"), "--device", "cpu",
                         "--output", str(root / "output")]
            with patch("sys.argv", arguments), \
                    patch.object(evaluate, "create_model", return_value=CaptureMeasurement()), \
                    patch.object(evaluate, "load_reference", return_value={}):
                evaluate.main()
        self.assertEqual(len(measurements), 1)
        np.testing.assert_array_equal(measurements[0][0, 0], expected)
        self.assertEqual(mask_strides, [((1, 16), (1, 16))])


if __name__ == "__main__":
    unittest.main()
