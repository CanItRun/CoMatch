from torch import nn
import torch
from copy import deepcopy
from typing import TypeVar

Module = TypeVar('Module', bound=nn.Module)


def EMA(model: Module, alpha=0.999) -> Module:
    """
    Exponential Moving Average(EMA) for nn.Module
    Args:
        model: nn.Module, An EMA wrapper of original model
        alpha: float, default 0.999, decay ratio of EMA

    Returns:
        A new cloned model that has a new method 'step'

    Notes:
        ema model will not generate gradient in its forward process
    """
    ema_model = deepcopy(model)

    [i.requires_grad_(False) for i in ema_model.parameters()]
    param_keys = set([k for k, _ in ema_model.named_parameters()])
    buffer_keys = set([k for k, _ in ema_model.named_buffers()])  # for Norm layers

    def step(alpha_=None):

        if alpha_ is None:
            alpha_ = alpha

        with torch.no_grad():
            for (k, ema_param), (_, param) in zip(ema_model.state_dict().items(), model.state_dict().items()):
                if k in param_keys:
                    ema_param.data.copy_(alpha_ * ema_param + (1 - alpha_) * param)
                elif k in buffer_keys:
                    ema_param.data.copy_(param)

    forward_ = ema_model.forward

    def forward(*args, **kwargs):
        with torch.no_grad():
            return forward_(*args, **kwargs)

    ema_model.forward = forward
    ema_model.step = step
    ema_model.eval()
    return ema_model
