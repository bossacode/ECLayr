import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math
from eclayr.alpha._ecc.ecc import AlphaEccBackbone


class Ecc(Function):
    @staticmethod
    def forward(ctx, x, func):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, P, D].
            func (function or method): Function that calculates ECC and gradient if needed.

        Returns:
            torch.Tensor: Tensor of shape [B, steps].
        """
        backprop = x.requires_grad
        device = x.device
        ecc, grad = func(x.cpu().numpy(), backprop)
        if backprop:
            ctx.save_for_backward(torch.from_numpy(grad).to(device))
            ctx.input_size = x.shape
        return torch.from_numpy(ecc).to(device)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """_summary_

        Args:
            grad_out (torch.Tensor): Gradient w.r.t. to output. Tensor of shape [B, steps].

        Returns:
            _type_: _description_
        """
        grad_in = None
        if ctx.needs_input_grad[0]:
            grad_local, = ctx.saved_tensors                                         # shape: [B, P, D, steps]
            grad_in = torch.matmul(grad_local, grad_out.unsqueeze(1).unsqueeze(-1)) # shape: [B, P, D, 1]
        return grad_in.squeeze(-1), None


class AlphaEcc(nn.Module):
    def __init__(self, interval=[0, 1], steps=32, beta=0.01, *args, **kwargs):
        """
        Args:
            steps (int, optional):  Number of discretized points. Defaults to 32.
            beta (float, optional): Controls the magnitude of impulse that approximates the dirac delta function used for backpropagation. Smaller values yield higher impulse. Defaults to 0.1.
        """
        super().__init__()
        impulse = 1 / (abs(beta) * math.sqrt(math.pi))
        alphaecc = AlphaEccBackbone(interval, steps, impulse)
        self.cal_ecc = alphaecc.cal_ecc
        
    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Tensor of shape [B, C, steps].
        """
        return Ecc.apply(x, self.cal_ecc)


class GThetaEC(nn.Module):
    def __init__(self, gtheta_cfg=[32, 32]):
        """
        Args:
            features (list, optional): List containing the size of each layer. Defaults to [32, 32].
        """
        super().__init__()
        self.gtheta = self._make_gtheta(gtheta_cfg)

    def forward(self, x):
        return self.gtheta(x)

    @staticmethod
    def _make_gtheta(gtheta_cfg):
        """
        Args:
            in_features:
            hidden_features:
        """
        num_layers = len(gtheta_cfg) - 1
        layer_list = []
        for i in range(num_layers):
            layer_list.append(nn.Linear(gtheta_cfg[i], gtheta_cfg[i+1]))
            layer_list.append(nn.ReLU())
        return nn.Sequential(*layer_list)


class ECLayr(nn.Module):
    def __init__(self, interval, steps, gtheta_cfg, beta=0.01, *args, **kwargs):
        """_summary_

        Args:
            steps (int, optional): _description_. Defaults to 32.
            gtheta_cfg (list, optional): _description_. Defaults to [32, 32].
            beta (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.ecc_layer = AlphaEcc(interval, steps, beta)
        self.gtheta = self._make_gtheta(gtheta_cfg)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W]

        Returns:
            output: Tensor of shape [B, gtheta_cfg[-1]]
        """
        ecc = self.ecc_layer(x)     # shape: [B, steps]
        return self.gtheta(ecc)     # shape: [B, gtheta_cfg[-1]]
    
    @staticmethod
    def _make_gtheta(gtheta_cfg):
        layer_list = []
        layer_list.append(nn.Linear(gtheta_cfg[0], gtheta_cfg[1]))
        num_layers = len(gtheta_cfg) - 1
        for i in range(1, num_layers):
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Linear(gtheta_cfg[i], gtheta_cfg[i+1]))
            # layer_list.append(nn.BatchNorm1d(gtheta_cfg[i+1]))
        return nn.Sequential(*layer_list)    
