import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def argmax_one_hot(x: torch.Tensor, dim: int):
    max_values, _ = x.max(dim=dim, keepdim=True)
    return torch.where(
        condition=x == max_values,
        input=torch.Tensor([1.0]).to(dtype=x.dtype, device=x.device),
        other=torch.Tensor([0.0]).to(dtype=x.dtype, device=x.device),
        out=x,
    )  # potentially make it the value itself? torch.where(x == max_values, x, 0.0)


def get_init_weight(shape, fan_in, init_type, scale, dtype=torch.float32):
    if init_type == "kaiming_uniform":
        return init_kaiming_uniform(
            shape=shape, fan_in=fan_in, scale=scale, dtype=dtype
        )
    elif init_type == "truncated_normal":
        return init_truncated_normal(
            shape=shape, fan_in=fan_in, scale=scale, dtype=dtype
        )
    else:
        raise ValueError(f"Unknown init_type: {init_type}")


def init_kaiming_uniform(shape, fan_in, scale, dtype=torch.float32):
    range_ = scale * (3 / fan_in) ** 0.5
    return torch.zeros(shape, dtype=dtype).uniform_(-range_, range_)


def init_truncated_normal(shape, fan_in, scale, dtype=torch.float32):
    std = (scale / fan_in) ** 0.5
    low = -2 * scale
    high = 2 * scale
    t = torch.zeros(shape, dtype=dtype)
    return trunc_normal_(t, mean=0.0, std=std, a=low, b=high)


def get_init_bias(shape, fan_in=None, fan_out=None, dtype=torch.float32):
    if fan_in is not None:
        raise ValueError("fan_in unsupported")
    if fan_out is not None:
        raise ValueError("fan_out unsupported")
    return torch.zeros(shape, dtype=dtype)


def stable_softmax_temperature(x: torch.Tensor, temperature: float, dim: int = -1) -> torch.Tensor:
    return F.softmax(x / temperature, dim=dim)
