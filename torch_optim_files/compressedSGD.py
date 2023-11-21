import torch
from torch import Tensor
from .optimizer import (Optimizer, required, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc)
from typing import List, Optional, Tuple
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
import math
import numpy as np


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
    def __init__(self, alpha: float):
        assert alpha > 0, 'Number of transmitted coordinates must be positive'
        self.alpha = alpha

    def getK(self, tensor: Tensor) -> int:
        result = int(tensor.numel() * self.alpha)
        assert result > 0, 'Number of transmitted coordinates must be positive'
        return result
 
    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        k = self.getK(tensor)
        tensor *= getTopKMask(tensor, k)
        return (tensor, k)


class RandKCompressor(Compression):
    def __init__(self, alpha: float):
        assert alpha > 0, 'Number of transmitted coordinates must be positive'
        self.alpha = alpha

    def getK(self, tensor: Tensor) -> int:
        result = int(tensor.numel() * self.alpha)
        assert result > 0, 'Number of transmitted coordinates must be positive'
        return result
 
    def compress(self, tensor: Tensor) -> Tuple[Tensor, int]:
        k = self.getK(tensor)
        mask = torch.zeros_like(tensor).index_fill_(
            0, torch.randperm(tensor.numel(), device='cuda:0')[:k], torch.tensor(1)
        )
        tensor *= mask
        return (tensor, k)


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
        c = np.random.binomial(size=1, n=1, p=p)[0]
        if c == 1 or not self.prevG:
            self.prevG = nabla
            self.prevNabla = nabla
            return nabla, self.dim
        result, k = self.compressor.compress(nabla - self.prevNabla)
        result += self.prevG
        self.prevNabla = nabla
        self.prevG = result
        return result, k



__all__ = ['SGD', 'sgd']

class compressedSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize: bool = False, foreach: Optional[bool] = None,
                 differentiable: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        self.compressor = MarinaCompressor(dim=15142970)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        shapes = list(map(lambda tensor: tensor.shape, d_p_list))
        numels = list(map(lambda tensor: tensor.numel(), d_p_list))
        stretched_tensors = list(map(lambda tensor: tensor.data.reshape(-1), d_p_list))
        long_tensor = torch.cat(stretched_tensors)

        long_tensor, self.last_coordinates_transmitted = self.compressor.compress(long_tensor)

        splitted_tensors = long_tensor.split(numels)
        for i, tensor in enumerate(splitted_tensors):
            d_p_list[i].data = tensor.reshape(shapes[i])


        return has_sparse_grad


    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
        else:
            foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


def _multi_tensor_sgd(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, momentum_buffer_list], with_indices=True)
    for ((device_params, device_grads, device_momentum_buffer_list), indices) in grouped_tensors.values():
        device_has_sparse_grad = any(grad.is_sparse for grad in device_grads)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        if momentum != 0:
            bufs = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = \
                            torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs

        if not device_has_sparse_grad:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)
