import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from cacti.data import Davis2017, load_mask
from cacti.models.spinsci import (SpinSCI, MCIRB, rgb_to_bayer_mosaic,
                                  add_bayer_mosaic_delta, init_mosaic_single_step,
                                  bayer_init_mosaic_single_step)
from cacti.release import create_model, load_reference, mask_normalizer, normalize_config, read_config
from cacti.training import GlobalBatchSampler, epoch_lr, reconstruction_loss
from cacti.utils.optim_builder import build_optimizer
from cacti.utils.utils import A

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class PaperEquations(unittest.TestCase):
    def test_couplings_and_mean_field_match_equations(self):
        torch.manual_seed(7)
        refiner = MCIRB(8, {"cond_dim": 3, "K": 3, "refine_hidden": 4,
                                    "lambda_phis": .7, "gamma": .8, "beta": .9})
        x, cond = torch.randn(2, 5, 8), torch.randn(2, 5, 3)
        logits = torch.randn(2, 2, 5, 5)
        pair_mask = torch.zeros(2, 5, 5)
        pair_mask[:, 0, 1] = pair_mask[:, 1, 0] = -100
        valid = pair_mask == 0
        penalty = sum(weight * (cond[:, :, i, None] - cond[:, None, :, i]).abs()
                      for i, weight in enumerate([1., 1., .7]))
        raw = (logits.mean(1)-penalty).masked_fill(~valid, 0)
        centered = (raw-raw.sum(-1, keepdim=True)/valid.sum(-1, keepdim=True)).masked_fill(~valid, 0)
        expected = .8 * .5 * (centered+centered.transpose(-1, -2))
        expected.diagonal(dim1=-2, dim2=-1).zero_()
        calls = []
        real_bmm = torch.bmm
        def capture(a, b):
            calls.append((a.detach().clone(), b.detach().clone()))
            return real_bmm(a, b)
        with patch("torch.bmm", side_effect=capture):
            output = refiner(x, logits, pair_mask, cond)
        self.assertEqual(len(calls), 3)
        state = torch.tanh(refiner.to_m0(torch.cat([x, cond], -1)))
        field = refiner.to_h(torch.cat([x, cond], -1))
        for coupling, actual_state in calls:
            torch.testing.assert_close(coupling, expected)
            torch.testing.assert_close(actual_state, state)
            state = torch.tanh(.9 * (field + expected @ state))
        torch.testing.assert_close(output, x)  # zero-initialized residual head
        self.assertLessEqual(state.abs().max().item(), 1)
        torch.nn.init.normal_(refiner.refine[-1].weight, std=.1)
        expected_output = x + torch.sigmoid(refiner.gate_scale * state) * refiner.refine(x)
        torch.testing.assert_close(refiner(x, logits, pair_mask, cond), expected_output)

    def test_initialization_and_conditioning_match_sensing(self):
        torch.manual_seed(23)
        phi = torch.randint(0, 2, (2, 8, 16, 18)).double()
        phi[..., 0, 0] = 0
        y = torch.rand(2, 1, 16, 18, dtype=torch.float64)
        normalizer = mask_normalizer(phi)
        backprojection = phi * y
        expected = backprojection + phi * ((y-(backprojection*phi).sum(1, keepdim=True))
                                           / (normalizer+1e-8))
        for initialize in (init_mosaic_single_step, bayer_init_mosaic_single_step):
            torch.testing.assert_close(initialize(y, phi, normalizer), expected)
        self.assertTrue(torch.equal(normalizer[..., 0, 0], torch.ones(2, 1, dtype=normalizer.dtype)))
        prior = create_model(read_config(ROOT/"configs/gray_cr8.json")).prior
        estimate = torch.rand_like(phi)
        estimate[0] = 0
        cond = prior._build_cond(phi, normalizer, estimate, 8, 9)
        nearest_mask = phi[..., ::2, ::2]
        normalized = estimate / estimate.abs().mean((1, 2, 3), keepdim=True).clamp_min(1e-8)
        # Bilinear resizing by exactly two averages each 2x2 neighborhood.
        average = lambda x: (x[..., 0::2, 0::2] + x[..., 0::2, 1::2]
                             + x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 4
        torch.testing.assert_close(cond[..., 0], nearest_mask)
        torch.testing.assert_close(cond[..., 1], average(normalized))
        torch.testing.assert_close(cond[..., 2], average(normalizer).expand(-1, 8, -1, -1))

    def test_bayer_selectors_and_adjoint(self):
        torch.manual_seed(11)
        rgb = torch.randn(2, 3, 8, 16, 18)
        delta = torch.randn(2, 8, 16, 18)
        original = torch.empty_like(delta)
        expected = rgb.clone()
        for channel, row, col in [(0, 0, 0), (1, 0, 1), (1, 1, 0), (2, 1, 1)]:
            original[:, :, row::2, col::2] = rgb[:, channel, :, row::2, col::2]
            expected[:, channel, :, row::2, col::2] += delta[:, :, row::2, col::2]
        torch.testing.assert_close(rgb_to_bayer_mosaic(rgb), original, rtol=0, atol=0)
        torch.testing.assert_close(add_bayer_mosaic_delta(rgb, delta), expected, rtol=0, atol=0)

    def test_mse_normalization_and_schedule(self):
        x, y = torch.randn(3, 3, 8, 4, 4), torch.randn(3, 3, 8, 4, 4)
        torch.testing.assert_close(reconstruction_loss(x, y), torch.nn.functional.mse_loss(x, y))
        self.assertEqual(epoch_lr(0, 200), 2e-4)
        self.assertEqual(epoch_lr(199, 200), 1e-6)
        self.assertTrue(all(epoch_lr(e, 200) >= epoch_lr(e+1, 200) for e in range(199)))

    def test_ddp_global_batch_and_gradient_equivalence(self):
        # Includes color batch 2 on four ranks, and non-divisible gray batch 10.
        for batch in (2, 10):
            samplers = [GlobalBatchSampler(40, batch, rank, 4) for rank in range(4)]
            shards = [list(iter(s))[0] for s in samplers]
            indices = [i for shard in shards for i, weight in shard if weight]
            self.assertEqual(len(indices), batch)
            self.assertEqual(len(set(indices)), batch)
            all_x = torch.arange(40, dtype=torch.float32)/40
            parameter = torch.tensor(.2, requires_grad=True)
            ((parameter*all_x[indices]-1)**2).mean().backward()
            expected = parameter.grad.clone()
            grads = []
            for shard in shards:
                p = torch.tensor(.2, requires_grad=True)
                values = all_x[[i for i, _ in shard]]
                weights = torch.tensor([w for _, w in shard])
                loss = reconstruction_loss((p*values)[:, None], torch.ones(len(shard), 1), weights)*4/batch
                loss.backward(); grads.append(p.grad)
            torch.testing.assert_close(torch.stack(grads).mean(), expected)

    def test_official_train_split_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"ImageSets/2017").mkdir(parents=True)
            (root/"ImageSets/2017/train.txt").write_text("train_scene\n")
            (root/"ImageSets/2017/val.txt").write_text("validation_scene\n")
            for scene in ("train_scene", "validation_scene"):
                directory = root/"JPEGImages/480p"/scene
                directory.mkdir(parents=True)
                for i in range(10):
                    cv2.imwrite(str(directory/f"{i:05}.jpg"), np.full((12, 16, 3), i, np.uint8))
            data = Davis2017(root, frames=8, crop=16, color=True)
            self.assertEqual(len(data), 3)
            self.assertTrue(all("train_scene" in str(window[0]) for window in data.windows))
            target, weight = data[0]
            self.assertEqual(tuple(target.shape), (3, 8, 16, 16))
            self.assertEqual(weight, 1)


class CheckpointCompatibility(unittest.TestCase):
    def test_constructor_defaults_match_reference_configs(self):
        for name, channels in (("gray", 1), ("color", 3)):
            with self.subTest(model=name):
                config = read_config(ROOT/f"configs/{name}_cr8.json")
                torch.manual_seed(19)
                direct = SpinSCI(color_channels=channels)
                torch.manual_seed(19)
                configured = create_model(config, "cudnn")
                self.assertTrue(direct.prior.fp32_io)
                self.assertEqual(list(direct.state_dict()), list(configured.state_dict()))
                self.assertEqual([n for n, _ in direct.named_parameters()],
                                 [n for n, _ in configured.named_parameters()])
                for name, value in direct.state_dict().items():
                    torch.testing.assert_close(value, configured.state_dict()[name], rtol=0, atol=0)

    def test_previous_model_name_and_saved_optimizer(self):
        for name in ("gray", "color"):
            with self.subTest(model=name):
                config = read_config(ROOT/f"configs/{name}_cr8.json")
                previous = json.loads(json.dumps(config))
                previous["model"]["type"] = "SpinSCIv4"
                self.assertEqual(normalize_config(previous), config)
                self.assertEqual(previous["model"]["type"], "SpinSCIv4")
                model = create_model(previous, "cudnn")
                checkpoint = torch.load(ROOT/f"checkpoints/spinsci_{name}_cr8.pth",
                                        map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                optimizer = build_optimizer(config["optimizer"], {"model": model})
                optimizer.load_state_dict(checkpoint["optim_state_dict"])
                for branch in (optimizer.muon, optimizer.adamw):
                    for parameter, state in branch.state.items():
                        for key, value in state.items():
                            if key != "step" and torch.is_tensor(value):
                                self.assertEqual(value.shape, parameter.shape)

    def test_all_reference_checkpoints(self):
        for item in json.loads((ROOT/"checkpoints/manifest.json").read_text()):
            with self.subTest(checkpoint=item["file"]):
                config = read_config(ROOT/item["config"])
                model = create_model(config, "cudnn").eval()
                metadata = load_reference(model, ROOT/"checkpoints"/item["file"])
                frames = config["model"]["frames"]
                phi = torch.randint(0, 2, (1, frames, 16, 16)).float()
                y = torch.rand(1, 1, 16, 16)
                with torch.inference_mode():
                    output = model(y, phi, mask_normalizer(phi))[-1]
                self.assertTrue(torch.isfinite(output).all())
                mosaic = rgb_to_bayer_mosaic(output) if output.ndim == 5 else output
                exposed = phi.sum(1, keepdim=True) > 0
                torch.testing.assert_close(A(mosaic, phi)[exposed], y[exposed], atol=2e-5, rtol=2e-5)

    def test_full_model_backward_and_optimizer_step(self):
        for name in ("gray", "color"):
            with self.subTest(model=name):
                torch.manual_seed(19)
                config = read_config(ROOT/f"configs/{name}_cr8.json")
                model = create_model(config, "cudnn").train()
                model.deep_supervision = False
                model.prior.aux_head.requires_grad_(False)
                optimizer = build_optimizer(config["optimizer"], {"model": model})
                shape = (1, 3, 8, 16, 16) if name == "color" else (1, 8, 16, 16)
                x = torch.rand(shape)
                phi = torch.randint(0, 2, (1, 8, 16, 16)).float()
                y = A(rgb_to_bayer_mosaic(x) if name == "color" else x, phi)
                before = model.prior.tail[-1].weight.detach().clone()
                loss = reconstruction_loss(model(y, phi, mask_normalizer(phi))[-1], x)
                loss.backward()
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    else:
                        self.assertIsNone(parameter.grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                self.assertFalse(torch.equal(before, model.prior.tail[-1].weight))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
