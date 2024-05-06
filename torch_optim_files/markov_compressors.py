import torch
from torch import Tensor

import math
import random

from abc import ABC, abstractmethod
from typing import Tuple


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
    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        pass


class NoneCompressor(BaseCompressor):
    def __init__(self, dim: int, device: str):
        self.dim = dim
        self.device = device

    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        return (tensor, tensor.numel())


class RandKCompressor(BaseCompressor):
    def __init__(self, dim: int, alpha: float, device: str):
        assert alpha > 0, 'Number of transmitted coordinates must be positive'
        self.dim = dim
        self.k = int(alpha * dim)
        self.device = device

    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        mask = torch.zeros_like(tensor).index_fill_(
            0, 
            torch.randperm(tensor.numel(), device=get_device(self.device))[:self.k],
            torch.tensor(1)
        )
        tensor *= mask
        return (tensor, self.k)


class MultCompressor():
    def __init__(self, dim: int, alpha: float, penalty: float, device: str):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.device = device
        self.probability = torch.full(
            (self.dim,), 1 / self.dim, dtype=torch.float, device=get_device(self.device)
        )
        self.penalty = penalty

    def compress(self, tensor):
        mask = getTopKMask(self.probability, self.k)
        self.probability = change_probability_multiplication(self.probability, mask, self.penalty)
        return tensor * mask, self.k


class SubtrCompressor():
    def __init__(self, dim: int, alpha: float, penalty: float, device: str):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.device = device
        self.probability = torch.full(
            (self.dim,), 1 / self.dim, dtype=torch.float, device=get_device(self.device)
        )
        self.penalty = penalty

    def compress(self, tensor):
        mask = getTopKMask(self.probability, self.k)
        self.probability = change_probability_subtraction(self.probability, mask, self.penalty)
        return tensor * mask, self.k


class ExpCompressor():
    def __init__(self, dim: int, alpha: float, beta: float, device: str):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.device = device
        self.penalty= torch.zeros(
            self.dim, dtype=torch.float, device=get_device(self.device)
        )
        self.beta = beta

    def compress(self, tensor):
        mask = getTopKMask(self.penalty, self.k)
        inv_mask = torch.ones_like(mask) - mask
        self.penalty = self.beta *self.penalty + (1 - self.beta) * inv_mask
        return tensor * mask, self.k

