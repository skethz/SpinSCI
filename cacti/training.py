"""MSE-only training helpers with exact effective batch sizes under DDP."""
import math

import torch
from torch.utils.data import Sampler


def epoch_lr(epoch, epochs, initial=2e-4, minimum=1e-6):
    """First epoch uses initial; the final epoch uses minimum."""
    if not 0 <= epoch < epochs:
        raise ValueError("epoch must be within the configured training run")
    progress = epoch / max(epochs-1, 1)
    return minimum + .5 * (initial-minimum) * (1 + math.cos(math.pi*progress))


def reconstruction_loss(prediction, target, sample_weights=None):
    """Eq. (MSE): average every T*C*H*W element, then average videos."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction {prediction.shape} != target {target.shape}")
    per_video = (prediction.float()-target.float()).square().flatten(1).mean(1)
    if sample_weights is None:
        return per_video.mean()
    return (per_video * sample_weights.to(per_video)).sum()


class GlobalBatchSampler(Sampler):
    """Shard exact global batches, including batches smaller than world size.

    Zero-weight padding makes every rank execute the same number of forwards;
    it does not enter the loss. SpinSCI contains no batch normalization.
    Incomplete final global batches are dropped explicitly.
    """
    def __init__(self, size, global_batch, rank=0, world_size=1, seed=2447):
        if global_batch < 1 or size < global_batch or not 0 <= rank < world_size:
            raise ValueError("Invalid dataset size, batch size, or distributed rank")
        self.size, self.global_batch = size, global_batch
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch = 0
        self.local_batch = math.ceil(global_batch/world_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.size // self.global_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for start in range(0, len(self)*self.global_batch, self.global_batch):
            group = order[start:start+self.global_batch]
            batch = []
            for offset in range(self.local_batch):
                position = self.rank*self.local_batch+offset
                valid = position < self.global_batch
                batch.append((group[position if valid else 0], float(valid)))
            yield batch
