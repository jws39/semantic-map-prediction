import torch
import torch.nn as nn
from torchvision import models
from models.networks.GNN import GNN, SpatialGNN


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_channel_in, n_class_out):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class_out, 1)

    def forward(self, input):
        B, T, C, cH, cW = input.shape
        input = input.view(B*T, C, cH, cW)

        x_original = self.conv_original_size0(input)  # [B*T, 64, 64, 64 ]
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)  # [B*T, 64, 32, 32 ]
        layer1 = self.layer1(layer0)  # [B*T, 64, 16, 16 ]
        layer2 = self.layer2(layer1)   # [B*T, 128, 8, 8 ]
        layer3 = self.layer3(layer2)  # [B*T, 256, 4, 4 ]
        layer4 = self.layer4(layer3)  # [B*T, 512, 2, 2 ]

        layer4 = self.layer4_1x1(layer4)  # [B*T, 512, 2, 2 ]
        x = self.upsample(layer4)  # [B*T, 512, 4, 4 ]

        layer3 = self.layer3_1x1(layer3)  # [B*T, 256, 4, 4 ]
        x = torch.cat([x, layer3], dim=1)  # [B*T, 768, 4, 4 ]
        x = self.conv_up3(x)   # [B*T, 512, 4, 4 ]

        x = self.upsample(x)  # [B*T, 512, 8, 8 ]
        layer2 = self.layer2_1x1(layer2)  # [B*T, 128, 8, 8 ]
        x = torch.cat([x, layer2], dim=1)  # [B*T, 640, 8, 8 ]
        x = self.conv_up2(x)  # [B*T, 256, 8, 8 ]

        x = self.upsample(x)  # [B*T, 256, 16, 16 ]
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)  # [B*T, 320, 16, 16 ]
        x = self.conv_up1(x)  # [B*T, 256, 16, 16 ]

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)  # [B*T, 128, 32, 32 ]

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  # [B*T, 64, 64, 64 ]

        out = self.conv_last(x)  # [B*T, 27, 64, 64 ]

        return out


class AE(nn.Module):
    def __init__(self, N, C, in_channels, inter_channels, out_channels, pool=(2, 2), factor=2):
        super().__init__()
        self.C = C
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(pool)
        self.pool_sp = nn.MaxPool2d((2, 2))
        self.linearKC = nn.Linear(inter_channels, C)
        self.linearNC = nn.Linear(N*N, C)
        self.gnn = GNN(inter_channels)
        self.spatialgnn = SpatialGNN(inter_channels, N, N)

        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        self.back = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, SP):
        '''x: bs, f, h, w
            SP: bs, C, h, w
        '''
        SP = torch.softmax(SP, dim=1)
        SP = self.pool(SP)  # bs, C, h/2, w/2
        SP = SP.reshape(SP.shape[0], SP.shape[1], -1)  # bs, C, n

        t = self.trans(x)  # bs, k, h/4, w/4
        y = t
        # y = self.pool(t)  # bs, k, h/2, w/2
        size = y.shape
        y = y.reshape(y.shape[0], y.shape[1], -1)  # bs, k, n

        # object
        # no object
        # A = torch.matmul(y.permute(0, 2, 1), y)

        # with object
        sigma = self.linearKC(self.linearNC(y).permute(0, 2, 1))  # bs, c, c
        A = torch.matmul(SP.permute(0, 2, 1), torch.matmul(sigma, SP))  # bs, n, n

        y = y.permute(0, 2, 1)  # bs, n, k

        # y = self.gnn(A, y) + y
        se_y = self.gnn(A, y) + y


        # no spatial
        # y = self.spatialgnn(y) + y
        sp_y = self.spatialgnn(y) + y

        # gate
        # y = self.gate(se_y, sp_y)
        y = se_y+sp_y

        y = self.dropout(self.up(y.permute(0, 2, 1).reshape(size))) + t
        y = self.back(y)
        return self.dropout(y)


class AM(nn.Module):
    def __init__(self, N, C, in_channels, inter_channels, out_channels, pool=(2, 2), factor=2):
        super().__init__()
        self.C = C
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(pool)
        self.pool_sp = nn.MaxPool2d((2, 2))
        self.linearKC = nn.Linear(inter_channels, C)
        self.linearNC = nn.Linear(N*N, C)
        self.gnn = GNN(inter_channels)
        self.spatialgnn = SpatialGNN(inter_channels, N, N)
        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        self.back = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, SP):
        '''x: bs, f, h, w
            SP: bs, C, h, w
        '''
        SP = torch.softmax(SP, dim=1)
        SP = self.pool(SP)  # bs, C, h/2, w/2
        SP = SP.reshape(SP.shape[0], SP.shape[1], -1)  # bs, C, n

        t = self.trans(x)  # bs, k, h/4, w/4
        y = t
        # y = self.pool(t)  # bs, k, h/2, w/2
        size = y.shape
        y = y.reshape(y.shape[0], y.shape[1], -1)  # bs, k, n

        # no object
        # A = torch.matmul(y.permute(0, 2, 1), y)

        sigma = self.linearKC(self.linearNC(y).permute(0, 2, 1))  # bs, c, c
        A = torch.matmul(SP.permute(0, 2, 1), torch.matmul(sigma, SP))  # bs, n, n

        y = y.permute(0, 2, 1)  # bs, n, k
        # y = self.gnn(A, y) + y
        sm_y = self.gnn(A, y) + y

        # y = self.spatialgnn(y) + y
        sp_y = self.spatialgnn(y) + y

        # gate
        # y = self.gate(sm_y, sp_y)
        y = sm_y + sp_y

        y = self.dropout(self.up(y.permute(0, 2, 1).reshape(size))) + t
        y = self.back(y)
        return self.dropout(y), A


class ResNetUNetDAMLastLayerv2(nn.Module):
    def __init__(self, n_channel_in, n_class_out):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer0_ae = AE(32, 27, 64, 64, 64, pool=(2, 2), factor=1)

        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_ae = AE(16, 27, 64, 64, 64, pool=(4, 4), factor=1)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer2_ae = AE(8, 27, 128, 128, 128, pool=(8, 8), factor=1)

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer3_ae = AE(4, 27, 256, 256, 256, pool=(16, 16), factor=1)

        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        self.layer4_ae = AE(2, 27, 512, 512, 512, pool=(32, 32), factor=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.up3_ae = AE(4, 27, 512, 512, 512, pool=(16, 16), factor=1)

        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.up2_ae = AE(8, 27, 256, 256, 256, pool=(8, 8), factor=1)

        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.up1_ae = AE(16, 27, 256, 256, 256, pool=(4, 4), factor=1)

        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.up0_ae = AE(32, 27, 128, 128, 128, pool=(2, 2), factor=1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.up0_last = AM(64, 27, 64, 64, 64, pool=(1, 1), factor=1)

        self.conv_last = nn.Conv2d(64, n_class_out, 1)

    def forward(self, input):
        B, T, C, cH, cW = input.shape
        input = input.view(B*T, C, cH, cW)

        x_original = self.conv_original_size0(input)  # [B*T, 64, 64, 64 ]
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)  # [B*T, 64, 32, 32 ]
        layer0 = self.layer0_ae(layer0, input)

        layer1 = self.layer1(layer0)  # [B*T, 64, 16, 16 ]
        layer1 = self.layer1_ae(layer1, input)

        layer2 = self.layer2(layer1)   # [B*T, 128, 8, 8 ]
        layer2 = self.layer2_ae(layer2, input)

        layer3 = self.layer3(layer2)  # [B*T, 256, 4, 4 ]
        layer3 = self.layer3_ae(layer3, input)

        layer4 = self.layer4(layer3)  # [B*T, 512, 2, 2 ]
        layer4 = self.layer4_ae(layer4, input)

        layer4 = self.layer4_1x1(layer4)  # [B*T, 512, 2, 2 ]
        x = self.upsample(layer4)  # [B*T, 512, 4, 4 ]

        layer3 = self.layer3_1x1(layer3)  # [B*T, 256, 4, 4 ]
        x = torch.cat([x, layer3], dim=1)  # [B*T, 768, 4, 4 ]
        x = self.conv_up3(x)   # [B*T, 512, 4, 4 ]
        x = self.up3_ae(x, input)

        x = self.upsample(x)  # [B*T, 512, 8, 8 ]
        layer2 = self.layer2_1x1(layer2)  # [B*T, 128, 8, 8 ]
        x = torch.cat([x, layer2], dim=1)  # [B*T, 640, 8, 8 ]
        x = self.conv_up2(x)  # [B*T, 256, 8, 8 ]
        x = self.up2_ae(x, input)

        x = self.upsample(x)  # [B*T, 256, 16, 16 ]
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)  # [B*T, 320, 16, 16 ]
        x = self.conv_up1(x)  # [B*T, 256, 16, 16 ]
        x = self.up1_ae(x, input)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)  # [B*T, 128, 32, 32 ]
        x = self.up0_ae(x, input)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  # [B*T, 64, 64, 64 ]
        x, up0_am = self.up0_last(x, input)

        out = self.conv_last(x)  # [B*T, 27, 64, 64 ]

        return out, up0_am


if __name__ == "__main__":

    x = torch.rand(4, 10, 27, 64, 64)
    x5 = torch.rand(4, 27, 128, 128)
    rgb = torch.rand(4, 10, 3, 128, 128)

    model = ResNetUNetDAMLastLayerv2(n_channel_in=27, n_class_out=27)


    # y = model(rgb)
    for i in range(5):
        y = model(x)
        print('y', y[0].shape)

