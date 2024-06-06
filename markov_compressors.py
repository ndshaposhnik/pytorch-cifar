import torch
from torch import Tensor

import math
import random

from abc import ABC, abstractmethod
from typing import Tuple, List


def apply_compressor(compressor, tensors: List[Tensor]): # функция меняет второй аргумент
    shapes = [tensor.shape for tensor in tensors]
    numels = [tensor.numel() for tensor in tensors]
    stretched_tensors = [tensor.data.reshape(-1) for tensor in tensors]
    long_tensor = torch.cat(stretched_tensors)

    long_tensor = compressor.compress(long_tensor)

    splitted_tensors = long_tensor.split(numels)
    for i, tensor in enumerate(splitted_tensors):
        tensors[i].data = tensor.reshape(shapes[i])


def get_device(device: str):
    if device == 'cuda':
        return 'cuda:0'
    elif device == 'cpu':
        return 'cpu'
    assert False, f'Unknown device: {device}'


def getTopKMask(tensor: Tensor, k: int) -> Tensor:
    absTensor = torch.abs(tensor)
    return torch.zeros_like(tensor).index_fill_(
        0, absTensor.topk(k).indices, torch.tensor(1)
    )

def getRandomMask(tensor: Tensor, k: int) -> Tensor:
    idxs = torch.multinomial(tensor, k, replacement=False)
    return torch.zeros_like(tensor).index_fill_(
        0, idxs, torch.tensor(1)
    )


def change_probability_multiplication(probability: Tensor, mask: Tensor, penalty: float) -> Tensor:
    assert probability.numel() == mask.numel(), 'probability and shape are not the same shape'
    n = probability.numel()
    k = mask.sum().item()
    assert k > 0, 'empty mask'
    inv_mask = torch.ones_like(mask) - mask
    sumReduced = torch.sum(mask * probability * (1 - penalty)).item()
    probability -= mask * probability * (1 - penalty)
    probability += inv_mask * sumReduced / (n - k)
    return probability


def change_probability_subtraction(probability, mask, penalty):
    assert probability.numel() == mask.numel(), 'probability and shape are not the same shape'
    n = probability.numel()
    k = mask.sum().item()
    assert k > 0, 'empty mask'

    inv_mask = torch.ones_like(mask) - mask
    tmp_probability = torch.clone(probability)
    tmp_probability -= mask * penalty
    tmp_probability = torch.maximum(tmp_probability, torch.zeros_like(tmp_probability))
    sumReduced = torch.sum(probability - tmp_probability).item()
    probability = tmp_probability
    probability += inv_mask * sumReduced / (n - k)
    return probability


class BaseCompressor(ABC):
    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def compress(self, tensor: Tensor) -> Tensor:
        pass


class NoneCompressor(BaseCompressor):
    def __init__(self, dim: int, device: str):
        self.dim = dim
        self.device = device
        self.k = dim

    def __str__(self):
        return 'None'

    def compress(self, tensor: Tensor) -> Tensor:
        return tensor


class RandKCompressor(BaseCompressor):
    def __init__(self, dim: int, alpha: float, device: str):
        assert alpha > 0, 'Number of transmitted coordinates must be positive'
        self.dim = dim
        self.k = int(alpha * dim)
        self.device = device

    def __str__(self):
        return 'RandK'

    def compress(self, tensor: Tensor) -> Tensor:
        mask = torch.zeros_like(tensor).index_fill_(
            0, 
            torch.randperm(tensor.numel(), device=get_device(self.device))[:self.k],
            torch.tensor(1)
        )
        tensor *= mask
        return tensor * self.dim / self.k


class MultCompressor():
    def __init__(self, dim: int, alpha: float, penalty: float, device: str, isDeterministic=False):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.device = device
        self.probability = torch.full(
            (self.dim,), 1 / self.dim, dtype=torch.float, device=get_device(self.device)
        )
        self.penalty = penalty  # p -> p * self.penalty 
        self.isDeterministic = isDeterministic

    def __str__(self):
        return 'Mult'

    def compress(self, tensor):
        mask = getTopKMask(self.probability, self.k)
        self.probability = change_probability_multiplication(self.probability, mask, self.penalty)
        return tensor * mask * self.dim / self.k


class SubtrCompressor():
    def __init__(self, dim: int, alpha: float, penalty: float, device: str, isDeterministic=False):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.device = device
        self.probability = torch.full(
            (self.dim,), 1 / self.dim, dtype=torch.float, device=get_device(self.device)
        )
        self.penalty = penalty
        self.isDeterministic = isDeterministic

    def __str__(self):
        return 'Subtr'

    def compress(self, tensor):
        if self.isDeterministic:
            mask = getTopKMask(self.probability, self.k)
        else:
            mask = getRandomMask(self.probability, self.k)

        self.probability = change_probability_subtraction(self.probability, mask, self.penalty)
        return tensor * mask * self.dim / self.k


class ExpCompressor():
    def __init__(self, dim: int, alpha: float, beta: float, device: str, isDeterministic=False):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.device = device
        self.penalty= torch.zeros(
            self.dim, dtype=torch.float, device=get_device(self.device)
        )
        self.beta = beta
        self.isDeterministic = isDeterministic

    def __str__(self):
        return 'Exp'

    def compress(self, tensor):
        if self.isDeterministic:
            mask = getTopKMask(self.probability, self.k)
        else:
            mask = getRandomMask(self.probability, self.k)

        inv_mask = torch.ones_like(mask) - mask
        self.penalty = self.beta * self.penalty + (1 - self.beta) * inv_mask
        return tensor * mask * self.dim / self.k


class BanLastMCompressor(BaseCompressor):
    def __init__(self, dim, alpha, M, device):
        super().__init__(dim)
        self.k = int(self.dim * alpha)
        self.M = M
        self.device = device
        self.history = torch.zeros(self.dim, dtype=torch.int, device=self.device)

    def __str__(self):
        return 'BanLastM'

    def _get_mask(self):
        zeros = (self.history == 0).nonzero().flatten()
        assert len(zeros) >= self.k
        indices = torch.randperm(len(zeros))[:self.k]
        assert len(indices) == self.k
        return torch.zeros_like(self.history).index_fill_(
            0, zeros[indices], torch.tensor(1),
        )

    def _update_history(self, mask):
        self.history -= torch.ones_like(self.history)
        self.history = torch.maximum(self.history, torch.zeros_like(self.history))
        self.history += mask * self.M

    def compress(self, tensor):
        mask = self._get_mask()
        self._update_history(mask)
        assert len(mask.nonzero()) > 0
        return tensor * mask * self.dim / self.k
