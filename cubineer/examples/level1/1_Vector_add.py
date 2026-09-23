import torch


class Model(torch.nn.Module):
    def forward(self, a, b):
        return a + b


def get_inputs():
    return [torch.randn(1024), torch.randn(1024)]


def get_init_inputs():
    return []
