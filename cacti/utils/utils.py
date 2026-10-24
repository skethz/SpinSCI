"""SCI forward operator and adjoint, inherited from CACTI/STFormer."""
import torch

def A(x, Phi):
    return torch.sum(x * Phi, dim=1, keepdim=True)

def At(y, Phi):
    return y * Phi
