from typing import Callable

import torch
import torch.nn.functional as F


def d_metric_pool(
    distance_metric: str, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    if distance_metric.lower() == "manhattan":
        return F.pairwise_distance(x, y, p=1.0)
    elif distance_metric.lower() == "cosine":
        return 1 - F.cosine_similarity(x, y)
    else:
        raise Exception(f"Distance metric <{distance_metric}> not implemented")


def activation_pool(activation: str) -> Callable:
    if activation.lower() == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif activation.lower() == "tanh":
        return torch.nn.functional.tanh
    else:
        raise Exception(f"Activation <{activation}> not implemented")
