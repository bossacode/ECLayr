import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from eclayr.cubical.cubeclayr import CubECLayr, CubDECC


# Cnn
class Cnn(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
            )
        self.fc = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )

    def forward(self, x):
        x, x_dtm = x
        x = F.relu(self.conv(x))
        x = self.fc(x.flatten(1))
        return x


# Cnn + ECLayr
class ECCnn_i(Cnn):
    def __init__(self, in_channels=1, num_classes=10, *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.eclayr = CubECLayr(postprocess=nn.Linear(kwargs["steps"], kwargs["topo_out"]), *args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(784 + kwargs["topo_out"], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )

    def forward(self, x):
        x, x_dtm = x
        ecc = F.relu(self.eclayr(x_dtm))    # ECLayr
        x = F.relu(self.conv(x))            # CNN
        x = torch.concat((x.flatten(1), ecc), dim=-1)
        x = self.fc(x)
        return x


# Cnn + ECLayr + ECLayr after conv
class ECCnn(Cnn):
    def __init__(self, in_channels=1, num_classes=10, *args, **kwargs):
        super().__init__(in_channels, num_classes)
        self.eclayr_1 = CubECLayr(interval=kwargs["interval_1"], sublevel=kwargs["sublevel_1"],
                                  postprocess=nn.Linear(kwargs["steps"], kwargs["topo_out"]), *args, **kwargs)
        self.eclayr_2 = CubECLayr(interval=kwargs["interval_2"], sublevel=kwargs["sublevel_2"],
                                  postprocess=nn.Linear(kwargs["steps"], kwargs["topo_out"]), *args, **kwargs)
        self.fc = nn.Sequential(
                    nn.Linear(784 + 2*kwargs["topo_out"], 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                    )

    def forward(self, x):
        x, x_dtm = x
        ecc_1 = F.relu(self.eclayr_1(x_dtm))    # first ECLayr
        x = F.relu(self.conv(x))                # CNN

        # second ECLayr after conv layer
        max_vals = x.amax(dim=(2, 3), keepdim=True)     # shape: (B, C, 1, 1)
        if (max_vals != 0).all():
            ecc_2 = F.relu(self.eclayr_2(x / max_vals)) # normalize between 0 and 1 for each data and channel
        else:
            ecc_2 = F.relu(self.eclayr_2(x))
        
        x = torch.concat((x.flatten(1), ecc_1, ecc_2), dim=-1)
        x = self.fc(x)
        return x


# Cnn + DECC + DECC after conv
class DECCnn(Cnn):
    def __init__(self, in_channels=1, num_classes=10, *args, **kwargs):
        super().__init__(in_channels, num_classes)
        # self.decc_1 = CubDECC(interval=kwargs["interval_1"], sublevel=kwargs["sublevel_1"], lam=kwargs["lam_1"], postprocess=nn.Linear(kwargs["steps_1"], topo_out_units), *args, **kwargs)
        self.eclayr_1 = CubECLayr(interval=kwargs["interval_1"], sublevel=kwargs["sublevel_1"],
                                  postprocess=nn.Linear(kwargs["steps"], kwargs["topo_out"]), *args, **kwargs)
        self.decc_2 = CubDECC(interval=kwargs["interval_2"], sublevel=kwargs["sublevel_2"],
                                  postprocess=nn.Linear(kwargs["steps"], kwargs["topo_out"]), *args, **kwargs)
        self.fc = nn.Sequential(
                    nn.Linear(784 + 2*kwargs["topo_out"], 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                    )

    def forward(self, x):
        x, x_dtm = x
        ecc_1 = F.relu(self.eclayr_1(x_dtm))    # first DECC (replaced with EClayr for computational reasons as the this layer does not require backpropagation)
        x = F.relu(self.conv(x))                # CNN

        # second DECC after conv layer
        max_vals = x.amax(dim=(2, 3), keepdim=True)     # shape: (B, C, 1, 1)
        if (max_vals != 0).all():
            ecc_2 = F.relu(self.decc_2(x / max_vals))   # normalize between 0 and 1 for each data and channel
        else:
            ecc_2 = F.relu(self.decc_2(x))
        
        x = torch.concat((x.flatten(1), ecc_1, ecc_2), dim=-1)
        x = self.fc(x)
        return x