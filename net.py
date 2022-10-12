import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset


def paper_init(weight, first_layer=False, omega=1):

    in_features = weight.shape[1] # input shape

    with torch.no_grad():
        if first_layer:
            bound = 1 / in_features # first layer [-1/in_features, 1/in_features] uniform distribution
        else:
            bound = np.sqrt(6 / in_features) / omega # rest of the layers [-sqrt(6/in_features)/omega, sqrt(6/in_features)/omega] uniform distribution

        weight.uniform_(-bound, bound)


class SineLayer(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            first_layer=False,
            omega=30,
            custom_init=None,
    ):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if custom_init is None:
            paper_init(self.linear.weight, first_layer=first_layer, omega=omega)
        else:
            custom_init_function_(self.linear.weight)

    def forward(self, x):

        return torch.sin(self.omega * self.linear(x)) # sin(omega * (Wx + b))

class ImageSiren(nn.Module):

    def __init__(
            self,
            hidden_features,
            hidden_layers=1,
            first_omega=30,
            hidden_omega=30,
            custom_init=None,
            ):
        super().__init__()
        in_features = 2
        out_features = 1

        net = []
        net.append(
                SineLayer(
                    in_features,
                    hidden_features,
                    first_layer=True,
                    custom_init=custom_init,
                    omega=first_omega,
            )
        )

        for _ in range(hidden_layers):
            net.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        first_layer=False,
                        custom_init=custom_init,
                        omega=hidden_omega,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)
        if custom_init is None:
            paper_init(final_linear.weight, first_layer=False, omega=hidden_omega)
        else:
            custom_init(final_linear.weight)

        net.append(final_linear)
        self.net = nn.Sequential(*net)


    def forward(self, x):

        return self.net(x)

class GradientUtils:
    @staticmethod
    def gradient(target, coords):
        return torch.autograd.grad(
            target, coords, grad_outputs=torch.ones_like(target), create_graph=True )[0]

    @staticmethod
    def divergence(grad, coords):
        div = 0.0
        for i in range(coords.shape[1]):
            div += torch.autograd.grad(
                grad[..., i], coords, torch.ones_like(grad[..., i]), create_graph=True,
            )[0][..., i : i + 1]
        return div

    @staticmethod
    def laplace(target, coords):
        grad = GradientUtils.gradient(target, coords)
        return GradientUtils.divergence(grad, coords)