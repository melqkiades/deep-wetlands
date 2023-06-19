
import torch
from torch import nn


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_dim=64, num_blocks=4):
        super().__init__()

        feature_list = [init_dim * (2 ** stair) for stair in range(num_blocks)]
        self.downs = nn.ModuleList()
        for idx, feature in enumerate(feature_list):
            self.downs.append(self.block(in_channels, feature))
            print(idx, in_channels, feature)
            in_channels = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        print(4, feature_list[-1], feature_list[-1] * 2)
        self.bottleneck = self.block(feature_list[-1], feature_list[-1] * 2)

        self.ups = nn.ModuleList()
        for idx, feature in enumerate(reversed(feature_list)):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(self.block(feature * 2, feature))
            print(idx, feature * 2, feature)

        self.conv = nn.Conv2d(
            in_channels=feature_list[0], out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return torch.sigmoid(self.conv(x))

    def block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
