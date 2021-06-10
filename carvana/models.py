import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    '''Performing same conv instead of valid conv and using batch norm'''

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    '''
    Source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py
    Same architecture as original UNet paper except expansive path has same H,W as contracting path
    '''

    def __init__(self, in_channels=3, out_channels=1, channels=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in channels:
            self.down.append(DoubleConv(in_channels, channel))
            in_channels = channel
        
        self.bottleneck = DoubleConv(channels[-1], channels[-1]*2)

        for channel in channels[::-1]:
            self.up.append(nn.ConvTranspose2d(channel*2, channel, kernel_size=2, stride=2))
            self.up.append(DoubleConv(channel*2, channel))

        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.up), 2):
            x = self.up[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up[idx+1](concat_skip)

        return self.final_conv(x)

class RotNet(nn.Module):
    '''Same architecture as the encoder of the UNet'''

    def __init__(self, in_channels=3, channels=[64, 128, 256, 512], num_classes=4):
        super(RotNet, self).__init__()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in channels:
            self.down.append(DoubleConv(in_channels, channel))
            in_channels = channel
        
        self.bottleneck = DoubleConv(channels[-1], channels[-1]*2)
        self.fc = nn.Linear(1024*16*16, num_classes) # assuming input size of 256x256

    def forward(self, x):
        for down in self.down:
            x = down(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        x = x.view(-1, 1024*16*16)
        x = self.fc(x)
        return x

def test():
    rot_net = RotNet(in_channels=3)
    rot_net_keys = set(rot_net.state_dict().keys())
    unet = UNet(in_channels=3, out_channels=1)
    unet_keys = set(unet.state_dict().keys())
    print(unet_keys.intersection(rot_net_keys))

    # x = torch.randn((8, 3, 256, 256))
    # model = UNet(in_channels=1, out_channels=3)
    # preds = model(x)
    # print(preds.shape)
    # print(x.shape)

if __name__ == "__main__":
    test()