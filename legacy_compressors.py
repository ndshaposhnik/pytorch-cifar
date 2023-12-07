import torch
from torch import Tensor

import math
import numpy as np
import random

def getTopKMask(tensor: Tensor, k: int) -> Tensor:
    absTensor = torch.abs(tensor)
    return torch.zeros_like(tensor).index_fill_(
        0, absTensor.topk(k).indices, torch.tensor(1)
    )


class Compression:
    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        pass


class NoneCompressor(Compression):
    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        return (tensor, tensor.numel())


class TopKCompressor(Compression):
    def __init__(self, dim: int, alpha: float):
        assert alpha > 0, 'Number of transmitted coordinates must be positive'
        self.dim = dim
        self.k = int(alpha * dim)

    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        tensor *= getTopKMask(tensor, self.k)
        return (tensor, self.k)


class RandKCompressor(Compression):
    def __init__(self, dim: int, alpha: float):
        assert alpha > 0, 'Number of transmitted coordinates must be positive'
        self.dim = dim
        self.k = int(alpha * dim)

    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        mask = torch.zeros_like(tensor).index_fill_(
            0, torch.randperm(tensor.numel(), device='cuda:0')[:k], torch.tensor(1)
        )
        tensor *= mask
        return (tensor, self.k)


class TopUnknownCompressor(Compression):
    def __init__(self, beta: float):
        assert beta > 0, 'Number of transmitted coordinates must be positive'
        self.beta = beta

    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        dim = tensor.numel()
        bound = self.beta * tensor.norm() / math.sqrt(dim)
        mask = torch.abs(tensor) >= bound
        nonzero_coords = mask.sum()
        if nonzero_coords == 0:
            mask = torch.zeros_like(tensor).index_fill_(
              0, torch.abs(tensor).topk(1).indices, torch.tensor(1)
            )
            nonzero_coords = mask.sum()
        return (tensor * mask, nonzero_coords.item())


class ReduceProbabilityCompression(Compression):
    def __init__(self, dim, alpha=0.5, penalty: float = 0.5):
        self.dim = dim
        self.penalty = penalty
        self.probability = torch.full((self.dim,), 1 / self.dim, dtype=torch.float, device='cuda:0')
        self.k = int(self.dim * alpha)

    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        mask = getTopKMask(tensor * self.probability, self.k)
        inv_mask = torch.ones_like(mask) - mask
        probability = torch.softmax(tensor, dim=0)
        sumReduced = torch.sum(mask * self.probability * (1 - self.penalty)).item()
        probability -= mask * self.probability * (1 - self.penalty)
        probability += inv_mask * sumReduced / (tensor.shape[0] - self.k)
        return tensor * mask, self.k


class PenaltyCompressor(Compression):
    def __init__(self, dim, alpha=0.5, dropsTo=0.0, step=0.25):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.dropsTo = dropsTo
        self.step = step
        self.penalty = torch.full((self.dim,), 1 / self.dim, dtype=torch.float, device='cuda:0')

    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        mask = getTopKMask(tensor * self.penalty, self.k)
        self.penalty += self.step * torch.ones_like(self.penalty)
        self.penalty = torch.minimum(self.penalty, torch.ones_like(self.penalty))
        inv_mask = torch.ones_like(mask) - mask
        self.penalty = inv_mask * self.penalty + self.dropsTo * mask
        return tensor * mask, self.k


class MarinaCompressor(Compression):
    def __init__(self, dim, p = 0.5, compressor=TopKCompressor(alpha=0.5)):
        self.dim = dim
        self.p = p
        self.compressor = compressor
        self.prevG = None
        self.prevNabla = None

    def compress(self, nabla: Tensor) -> Tuple[Tensor, int]:
        c = np.random.binomial(size=1, n=1, p=self.p)[0]
        if c == 1 or self.prevG == None:
            self.prevG = nabla
            self.prevNabla = nabla
            return nabla, self.dim
        result, k = self.compressor.compress(nabla - self.prevNabla)
        result += self.prevG
        self.prevNabla = nabla
        self.prevG = result
        return result, k


def change_probability_multiplication(probability: Tensor, mask: Tensor, penalty: float) -> Tensor:
    assert probability.numel() == mask.numel(), 'probability and shape are not the same shape'
    n = probability.numel()
    k = mask.sum().item()
    assert k > 0, 'empty mask'
    inv_mask = torch.ones_like(mask) - mask
    sumReduced = torch.sum(mask * probability * penalty).item()
    probability -= mask * probability * penalty
    probability += inv_mask * sumReduced / (n - k)
    return probability


class MultiplicationPenaltyCompressor():
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty

    def compress(self, tensor):
        mask = getTopKMask(self.probability, self.k)
        self.probability = change_probability_multiplication(self.probability, mask, self.penalty)
        return tensor * mask, self.k
