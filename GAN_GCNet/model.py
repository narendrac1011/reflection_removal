import torch.nn as nn
import torch
from requirements import Interpolate

# A GCVGGBlock
class GCBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, activation_function=nn.LeakyReLU(negative_slope=0.01,inplace=True)):

        super(GCBlock, self).__init__()

        # Sequence of operations (nn.Sequential) consisting of convolutional layers, batch normalization, and activation functions
        # Structure of the block
        self.model = nn.Sequential(

            # Convolutional Layer
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),

            # Batch Normalization Layer
            nn.BatchNorm2d(middle_channels),

            # Leaky ReLU Layer
            activation_function,

            # Convolutional Layer
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),

            # Batch Normalization Layer
            nn.BatchNorm2d(out_channels),

            # Leaky ReLU Layer
            activation_function
        )

    # Forward pass method for the block
    def forward(self, x):
        out = self.model(x)
        return out

# Architecture of the GCNet Generator
class GCNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GCNet, self).__init__()

        # The number of filters for each layer
        filter = [32, 64, 128, 256, 512]

        # Max-pooling and upsampling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.up = Interpolate(scale_factor=2, mode='bilinear')

        # GCBlock with different parameters for each layer of the network
        self.conv0_0 = GCBlock(in_channels, filter[0], filter[0])
        self.conv1_0 = GCBlock(filter[0], filter[1], filter[1])
        self.conv2_0 = GCBlock(filter[1], filter[2], filter[2])
        self.conv3_0 = GCBlock(filter[2], filter[3], filter[3])
        self.conv4_0 = GCBlock(filter[3], filter[4], filter[4])

        self.conv0_1 = GCBlock(filter[0] + filter[1], filter[0], filter[0])
        self.conv1_1 = GCBlock(filter[1] + filter[2], filter[1], filter[1])
        self.conv2_1 = GCBlock(filter[2] + filter[3], filter[2], filter[2])
        self.conv3_1 = GCBlock(filter[3] + filter[4], filter[3], filter[3])

        self.conv0_2 = GCBlock(filter[0]*2 + filter[1], filter[0], filter[0])
        self.conv1_2 = GCBlock(filter[1]*2 + filter[2], filter[1], filter[1])
        self.conv2_2 = GCBlock(filter[2]*2 + filter[3], filter[2], filter[2])

        self.conv0_3 = GCBlock(filter[0]*3 + filter[1], filter[0], filter[0])
        self.conv1_3 = GCBlock(filter[1]*3 + filter[2], filter[1], filter[1])

        self.conv0_4 = GCBlock(filter[0]*4 + filter[1], filter[0], filter[0])

        # The final convolutional layers for producing the output of the network
        self.final1 = nn.Sequential(
            nn.Conv2d(filter[0], out_channels, kernel_size=3, padding=1),
            )
        self.final2 = nn.Sequential(
            nn.Conv2d(filter[0], out_channels, kernel_size=3, padding=1),
            )
        self.final3 = nn.Sequential(
            nn.Conv2d(filter[0], out_channels, kernel_size=3, padding=1),
            )
        self.final4 = nn.Sequential(
            nn.Conv2d(filter[0], filter[0], 5, padding=2),
            nn.BatchNorm2d(filter[0]),
            nn.LeakyReLU(negative_slope = 0.01,inplace = True),
            nn.Conv2d(filter[0], out_channels, kernel_size=3, padding=1),
            )

        # Initialize convolutional layers for gradient computation, i.e. for reflection removal
        self.G_x_D = nn.Conv2d(3, 1, kernel_size = 3, stride = 1, padding = 0, bias=False)
        self.G_y_D = nn.Conv2d(3, 1, kernel_size = 3, stride = 1, padding = 0, bias=False)
        self.G_x_G = nn.Conv2d(3, 1, kernel_size = 3, stride = 1, padding = 0, bias=False)
        self.G_y_G = nn.Conv2d(3, 1, kernel_size = 3, stride = 1, padding = 0, bias=False)

    # Forward pass method for the network
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output4 = self.final4(x0_4)
        return output4