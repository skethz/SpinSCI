"""External DAVIS2017 training frames and MATLAB/NumPy array readers.

Training videos and benchmark measurement files are supplied separately.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


def read_array(path, key):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return data[key]
    try:
        return scipy.io.loadmat(path)[key]
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as data:
            # MATLAB v7.3 stores dimensions in reverse order.
            return np.asarray(data[key]).transpose()


def load_mask(path, frames, key="mask", layout="HWT", crop=None):
    values = np.asarray(read_array(path, key), dtype=np.float32)
    if layout == "HWT":
        values = values.transpose(2, 0, 1)
    elif layout != "THW":
        raise ValueError("mask layout must be HWT or THW")
    if values.ndim != 3 or values.shape[0] != frames:
        raise ValueError(f"Expected {frames} mask frames, got {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Mask must contain finite nonnegative values")
    if crop is not None:
        h, w = values.shape[-2:]
        if min(h, w) < crop:
            raise ValueError(f"Mask {h}x{w} is smaller than crop {crop}")
        y, x = (h-crop)//2, (w-crop)//2
        values = values[:, y:y+crop, x:x+crop]
    return torch.from_numpy(np.ascontiguousarray(values))


class Davis2017(Dataset):
    """Contiguous T-frame windows, restricted to the official 2017 train list.

    Each window occurs once per epoch before shuffling. All frames in a window
    receive the same resize, crop, horizontal flip, and vertical flip. Images
    smaller than the crop are resized preserving aspect ratio first.
    """
    def __init__(self, root, frames=8, crop=256, color=False, resolution="480p"):
        root = Path(root)
        split = root / "ImageSets/2017/train.txt"
        if not split.is_file():
            raise FileNotFoundError(f"Official DAVIS2017 train list required: {split}")
        self.frames, self.crop, self.color = frames, crop, color
        self.windows = []
        for sequence in split.read_text().splitlines():
            sequence = sequence.strip()
            if not sequence:
                continue
            directory = root / "JPEGImages" / resolution / sequence
            paths = sorted(directory.glob("*.jpg"))
            if len(paths) < frames:
                raise ValueError(f"Insufficient training frames in {directory}")
            self.windows.extend(tuple(paths[i:i+frames])
                                for i in range(len(paths)-frames+1))
        if not self.windows:
            raise ValueError("The training split is empty")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        weight = 1.0
        if isinstance(index, tuple):
            index, weight = index
        images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in self.windows[index]]
        if any(image is None for image in images):
            raise ValueError(f"Cannot decode training window {index}")
        h, w = images[0].shape[:2]
        if any(image.shape[:2] != (h, w) for image in images):
            raise ValueError("All frames in a window must have the same dimensions")
        scale = max(1.0, self.crop / min(h, w))
        if scale > 1:
            h, w = math.ceil(h*scale), math.ceil(w*scale)
            images = [cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                      for im in images]
        y = int(torch.randint(h-self.crop+1, (1,)))
        x = int(torch.randint(w-self.crop+1, (1,)))
        flip_x, flip_y = bool(torch.rand(()) < .5), bool(torch.rand(()) < .5)
        targets = []
        for image in images:
            image = image[y:y+self.crop, x:x+self.crop]
            if flip_x:
                image = image[:, ::-1]
            if flip_y:
                image = image[::-1]
            if self.color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            else:
                # Same luminance conversion as CACTI's gray data pipeline.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0]
            targets.append(image.astype(np.float32) / 255.0)
        target = np.stack(targets, axis=1 if self.color else 0)
        return torch.from_numpy(np.ascontiguousarray(target)), float(weight)
