import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), use_relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_features, out_features, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_features)
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU()

    def forward(self, input):
        x = self.bn(self.conv(input))
        if self.use_relu:
            x = self.relu(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1, head_conv=1, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.conv1 = ConvBlock(in_features, out_features, (head_conv, 1, 1), padding=(1 if head_conv == 3 else 0, 0, 0))
        self.conv2 = ConvBlock(out_features, out_features, (1, 3, 3), (1, stride, stride), (0, 1, 1))
        self.conv3 = ConvBlock(out_features, out_features * 4, 1)
        if downsample:
            self.conv4 = ConvBlock(in_features, out_features * 4, 1, stride=(1, stride, stride), use_relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        x = residual = input
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample:
            residual = self.conv4(residual)
        x += residual
        return self.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, lrs=1, head_conv=1):
        super(ResidualBlock, self).__init__()
        self.init_block = BottleneckBlock(in_features, out_features, head_conv=head_conv, downsample=True)
        self.blocks = []
        for _ in range(1, lrs):
            self.blocks.append(BottleneckBlock(4 * out_features, out_features, head_conv=head_conv))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, input):
        x = self.init_block(input)
        for b in self.blocks:
            x = b(x)
        return x


class SlowFast(nn.Module):
    def __init__(self, layers=None, class_num=10, dropout=0.5):
        super(SlowFast, self).__init__()
        if layers is None:
            layers = [3, 4, 6, 3]

        self.fast_conv1 = ConvBlock(3, 8, (5, 7, 7), (1, 2, 2), (2, 3, 3))
        self.fast_maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.fast_res1 = ResidualBlock(8, 8, layers[0], 3)
        self.fast_res2 = ResidualBlock(32, 16, layers[1], 3)
        self.fast_res3 = ResidualBlock(64, 32, layers[2], 3)
        self.fast_res4 = ResidualBlock(128, 64, layers[3], 3)
        self.fast_avgpool = nn.AdaptiveAvgPool3d(1)

        self.lateral_conv1 = nn.Conv3d(8, 16, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.lateral_conv2 = nn.Conv3d(32, 64, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.lateral_conv3 = nn.Conv3d(64, 128, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.lateral_conv4 = nn.Conv3d(128, 256, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)

        self.slow_conv1 = ConvBlock(3, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3))
        self.slow_maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.slow_res1 = ResidualBlock(80, 64, layers[0], 3)
        self.slow_res2 = ResidualBlock(320, 128, layers[1], 3)
        self.slow_res3 = ResidualBlock(640, 256, layers[2], 3)
        self.slow_res4 = ResidualBlock(1280, 512, layers[3], 3)
        self.slow_avgpool = nn.AdaptiveAvgPool3d(1)

        self.dp = nn.Dropout3d(dropout)
        self.fc = nn.Linear(2304, class_num, bias=False)

    def forward(self, input):
        fast_x = input[:, :, ::2]
        slow_x = input[:, :, ::16]

        fast_x = self.fast_conv1(fast_x)
        lx1 = fast_x = self.fast_maxpool(fast_x)
        lx2 = fast_x = self.fast_res1(fast_x)
        lx3 = fast_x = self.fast_res2(fast_x)
        lx4 = fast_x = self.fast_res3(fast_x)
        fast_x = self.fast_res4(fast_x)
        fast_x = self.fast_avgpool(fast_x)
        fast_x = fast_x.view(-1, fast_x.shape[1])

        lx1 = self.lateral_conv1(lx1)
        lx2 = self.lateral_conv2(lx2)
        lx3 = self.lateral_conv3(lx3)
        lx4 = self.lateral_conv4(lx4)

        slow_x = self.slow_conv1(slow_x)
        slow_x = self.slow_maxpool(slow_x)
        slow_x = torch.cat([slow_x, lx1], dim=1)
        slow_x = self.slow_res1(slow_x)
        slow_x = torch.cat([slow_x, lx2], dim=1)
        slow_x = self.slow_res2(slow_x)
        slow_x = torch.cat([slow_x, lx3], dim=1)
        slow_x = self.slow_res3(slow_x)
        slow_x = torch.cat([slow_x, lx4], dim=1)
        slow_x = self.slow_res4(slow_x)
        slow_x = self.slow_avgpool(slow_x)
        slow_x = slow_x.view(-1, slow_x.shape[1])

        x = torch.cat([slow_x, fast_x], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x
