import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from cacti.release import read_config
from tools import train


ROOT = Path(__file__).resolve().parents[1]


class EpochBoundaryResume(unittest.TestCase):
    def test_zero_worker_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "DAVIS"
            frames = data / "JPEGImages/480p/train_scene"
            frames.mkdir(parents=True)
            (data / "ImageSets/2017").mkdir(parents=True)
            (data / "ImageSets/2017/train.txt").write_text("train_scene\n")
            rng = np.random.default_rng(73)
            for i in range(11):
                cv2.imwrite(str(frames/f"{i:05d}.jpg"),
                            rng.integers(0, 256, (20, 22, 3), dtype=np.uint8))
            np.save(root / "mask.npy", rng.integers(0, 2, (16, 16, 8)).astype(np.float32))
            config = read_config(ROOT / "configs/gray_cr8.json")
            config["model"].update(dim=8, blocks=1, stem_rank=4, tail_rank=4)
            config["model"]["ising_cfg"]["refine_hidden"] = 8
            config["training"].update(epochs=2, crop_size=16, batch_size=2)
            (root / "config.json").write_text(json.dumps(config))
            arguments = ["train", str(root / "config.json"), "--davis-root", str(data),
                         "--mask", str(root / "mask.npy"), "--device", "cpu", "--workers", "0"]
            with patch("sys.argv", arguments + ["--output", str(root / "complete")]):
                train.main()

            class StopAfterFirstEpoch(Exception):
                pass

            original_lr = train.epoch_lr

            def stop_at_boundary(epoch, *args):
                if epoch == 1:
                    raise StopAfterFirstEpoch
                return original_lr(epoch, *args)

            with patch("sys.argv", arguments + ["--output", str(root / "resumed")]), \
                    patch.object(train, "epoch_lr", side_effect=stop_at_boundary):
                with self.assertRaises(StopAfterFirstEpoch):
                    train.main()
            with patch("sys.argv", arguments + ["--output", str(root / "resumed"),
                        "--resume", str(root / "resumed/latest.pth")]):
                train.main()
            complete = torch.load(root / "complete/latest.pth", weights_only=True)
            resumed = torch.load(root / "resumed/latest.pth", weights_only=True)

            def check_equal(a, b):
                if torch.is_tensor(a):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        check_equal(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        check_equal(x, y)
                else:
                    self.assertEqual(a, b)

            check_equal(complete, resumed)


if __name__ == "__main__":
    unittest.main()
