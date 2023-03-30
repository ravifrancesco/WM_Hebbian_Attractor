import torch
import torch.nn as nn
import torchmetrics.functional as F


def rnn_pool(
    architecture: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    nonlinearity: str,
    bias: bool = True,
):
    if architecture.lower() == "rnn":
        model = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
        )
    elif architecture.lower() == "lstm":
        model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
        )
    else:
        raise Exception(f"Model <{architecture}> not implemented")
    model.eval()
    return model


class TileRNN(nn.Module):
    def __init__(
        self,
        architecture: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.model = rnn_pool(
            architecture, input_size, hidden_size, num_layers, nonlinearity, bias
        )
        self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out, self.state = self.model(x, self.state)
            return out

    def reset_state(self) -> None:
        self.state = None

    def __repr__(self) -> str:
        return self.model.__repr__()
