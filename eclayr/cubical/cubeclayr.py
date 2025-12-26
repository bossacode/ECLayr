import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math
from eclayr.cubical._ecc.ecc import ECC, ECC3d
from eclayr.cubical._decc.decc import DECC, DECC3d


class ECCwrapper(Function):
    @staticmethod
    def forward(ctx, x, func, device):
        """
        Args:
            x (tensor of shape (B, C, H, W) or (B, C, D, H, W)): Batch of 2D or 3D input images.
            func (function or method): Function that computes the Euler characteristic curves (and gradients if necessary).
            device (str or torch.device): Device to which the output (and gradient) will be moved.

        Returns:
            ecc (tensor of shape (B, C, steps)): Euler characteristic curves for each image and channel.
        """
        backprop = x.requires_grad
        ecc, grad = func(x.cpu().numpy(), backprop)
        if backprop:
            ctx.save_for_backward(torch.from_numpy(grad).to(device))
            ctx.input_shape = x.shape
        return torch.from_numpy(ecc).to(device)
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """
        Args:
            grad_out (tensor of shape (B, C, steps)): Upstream gradient w.r.t. to the output data.

        Returns:
            (grad_in, None) (tuple): Downstream gradient w.r.t. to the input data.
        """
        grad_in = None
        if ctx.needs_input_grad[0]:
            grad_local, = ctx.saved_tensors                                     # shape: (B, C, H*W, steps) or (B, C, D*H*W, steps)
            grad_in = torch.einsum("...ij,...j->...i", grad_local, grad_out)    # shape: (B, C, H*W) or (B, C, D*H*W)
            grad_in = grad_in.view(*ctx.input_shape)                            # shape: (B, C, H, W) or (B, C, D, H, W)
        return grad_in, None, None


class CubECLayr(nn.Module):
    def __init__(self, interval=[0., 1.], steps=32, sublevel=True, beta=0.1, postprocess=nn.Identity(), device="cpu",
                *args, **kwargs):
        """
        Args:            
            interval (Iterable[float], optional): Interval of filtration values to be considered. Defaults to [0., 1.].
            steps (int, optional):  Number of discretized points. Defaults to 32.
            sublevel (bool, optional): Whether to use sublevel set filtration. If False, superlevel set filtration will be used. Defaults to True.
            beta (float or "auto", optional): Controls the magnitude of impulse that approximates the dirac delta function used for backpropagation. Defaults to 0.1.
            postprocess (nn.Module, optional): Postprocessing layer. Defaults to nn.Identity().
            device (str or torch.device): Device to which the output (and gradient) will be moved. Defaults to "cpu".
        """
        assert len(interval) == 2, "Interval must consist of two values."
        assert interval[1] > interval[0], "End point of the interval must be larger than the starting point."
        assert steps > 1, "Steps must be larger than 1."
        assert beta == "auto" or isinstance(beta, (float, int)), "Beta must be either a number or 'auto'."

        super().__init__()
        self.interval = interval if sublevel else [-i for i in reversed(interval)] # change interval when superlevel set filtration is used
        self.steps = steps
        self.sublevel = sublevel
        self.impulse = (steps-1) / (interval[1]-interval[0]) if beta == "auto" else 1 / (abs(beta)*math.sqrt(math.pi))  # equivalent to beta = 2delta_t / sqrt(pi) when "auto"
        self.postprocess = postprocess
        self.device = device
        self.func = None

    def forward(self, x):
        """
        Args:
            x (tensor of shape (B, C, H, W) or (B, C, D, H, W)): Batch of 2D or 3D input images.

        Returns:
            out (tensor of shape (B, T)): Postprocessed Euler characteristic curves. "T" denotes the output dimension of the postprocessing layer.
        """
        if self.func is None:   # lazily initialize when first batch is passed
            dim = len(x.shape[2:])
            if dim == 2:
                # function that computes the Euler characteristic curves (and gradients if necessary)
                self.func = ECC(x.shape[2:], self.interval, self.steps, self.impulse).cal_ecc
            elif dim == 3:
                self.func = ECC3d(x.shape[2:], self.interval, self.steps, self.impulse).cal_ecc
            else:
                raise ValueError("Only 2D and 3D images can be used as input.")

        x = x if self.sublevel else -x                      # apply sublevel set filtration on negative data when superlevel set filtration is used
        ecc = ECCwrapper.apply(x, self.func, self.device)   # shape: (B, C, steps)
        out = self.postprocess(ecc.flatten(1))
        return out


class CubDECC(nn.Module):
    def __init__(self, interval=[0., 1.], steps=32, sublevel=True, lam=200, postprocess=nn.Identity(), device="cpu",
                 *args, **kwargs):
        """
        Args:            
            interval (Iterable[float], optional): Interval of filtration values to be considered. Defaults to [0., 1.].
            steps (int, optional):  Number of discretized points. Defaults to 32.
            sublevel (bool, optional): Whether to use sublevel set filtration. If False, superlevel set filtration will be used. Defaults to True.
            lam (float, optional): Controls the tightness of sigmoid approximation. Defaults to 200.
            postprocess (nn.Module, optional): Postprocessing layer. Defaults to nn.Identity().
            device (str or torch.device): Device to which the output (and gradient) will be moved. Defaults to "cpu".
        """
        assert len(interval) == 2, "Interval must consist of two values."
        assert interval[1] > interval[0], "End point of the interval must be larger than the starting point."
        assert steps > 1, "Steps must be larger than 1."

        super().__init__()
        self.interval = interval if sublevel else [-i for i in reversed(interval)] # change interval when superlevel set filtration is used
        self.steps = steps
        self.sublevel = sublevel
        self.lam = lam
        self.postprocess = postprocess
        self.device = device
        self.func = None
        
    def forward(self, x):
        """
        Args:
            x (tensor of shape (B, C, H, W) or (B, C, D, H, W)): Batch of 2D or 3D input images.

        Returns:
            out (tensor of shape (B, T)): Postprocessed sigmoid-approximated Euler characteristic curves. "T" denotes the output dimension of the postprocessing layer.
        """
        if self.func is None:   # lazily initialize when first batch is passed
            dim = len(x.shape[2:])
            if dim == 2:
                # function that computes the Euler characteristic curves (and gradients if necessary)
                self.func = DECC(x.shape[2:], self.interval, self.steps, self.lam).cal_ecc
            elif dim == 3:
                self.func = DECC3d(x.shape[2:], self.interval, self.steps, self.lam).cal_ecc
            else:
                raise ValueError("Only 2D and 3D images can be used as input.")

        x = x if self.sublevel else -x                      # apply sublevel set filtration on negative data when superlevel set filtration is used
        decc = ECCwrapper.apply(x, self.func, self.device)  # shape: (B, C, steps)
        out = self.postprocess(decc.flatten(1))
        return out