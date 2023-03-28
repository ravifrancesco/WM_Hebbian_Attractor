import torch
import torch.nn as nn


class CVModel(nn.Module):
    def __init__(
        self,
        repo_or_dir: str,
        model: str,
        model_dir: str = None,
        output_layer: int = -1,
    ) -> None:
        super().__init__()
        if model_dir:
            torch.hub.set_dir(model_dir)
        self.model = torch.hub.load(repo_or_dir, model)
        self.mode = nn.Sequential(*list(self.model.children())[:output_layer])
        self.out_features = list(self.model.children())[-1].out_features
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __repr__(self) -> str:
        return self.model.__repr__()
