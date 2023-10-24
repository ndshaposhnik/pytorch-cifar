from .optimizer import Optimizer, params_t

class compressedSGD(Optimizer):
    def __init__(
        self,
        params: params_t,
        lr: float,
        momentum: float = ...,
        dampening: float = ...,
        weight_decay: float = ...,
        nesterov: bool = ...,
    ) -> None: ...
