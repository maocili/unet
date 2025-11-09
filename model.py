import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ContractingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, kernel_size=3),
        )

    def forward(self, x):
        return self.net(x)


class ExpansiveBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2),
        )

        self.double_conv = DoubleConv(
            out_channels * 2, out_channels, kernel_size=3)

    def forward(self, x, copped_features):
        x = self.upsampling(x)
        x = torch.cat([x, copped_features], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first_out_channcels: int = 64, steps: int = 4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_layer = DoubleConv(in_channels=in_channels,
                                      out_channels=first_out_channcels)
        self.downlayers = nn.ModuleList()
        self.uplayers = nn.ModuleList()
        self.final_layer = nn.Conv2d(
            first_out_channcels, out_channels, kernel_size=1)

        for i in range(0, steps):
            in_ch = first_out_channcels * 2**i
            out_ch = first_out_channcels * 2**(i+1)
            self.downlayers.append(ContractingBlock(
                in_channels=in_ch, out_channels=out_ch))

        for i in range(steps, 0, - 1):
            in_ch = first_out_channcels * 2 ** i
            out_ch = first_out_channcels * 2 ** (i - 1)
            self.uplayers.append(ExpansiveBlock(
                in_channels=in_ch, out_channels=out_ch))

    def forward(self, x):
        x_coppied = []

        x = self.input_layer(x)
        x_coppied.append(x)

        for down in self.downlayers:
            x = down(x)
            x_coppied.append(x)

        # skip least coppied feature
        x_coppied.pop()

        for up in self.uplayers:
            crop_x = x_coppied.pop()
            x = up(x, crop_x)

        x = self.final_layer(x)
        return x


# model = UNet(in_channels=1, out_channels=3)
# x = torch.randn((3, 1, 256, 256))
# y = model(x)
# assert y.shape == (3, 3, 256, 256)