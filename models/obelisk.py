import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class OBELISK(nn.Module):
    def __init__(self, full_res=None, corner=False):
        super(OBELISK, self).__init__()

        self.corner = corner
        if full_res is None:
            raise ValueError("param 'full_res' should be filled")
        self.BS = full_res[0]
        self.o_m = full_res[1] // 3  # H//3
        self.o_n = full_res[2] // 3  # W//3
        self.o_o = full_res[3] // 3  # D//3

        self.ogrid_xyz = F.affine_grid(torch.eye(3, 4).unsqueeze(0).expand(self.BS, 3, 4),
                                       size=[self.BS, 1, self.o_m, self.o_n, self.o_o],
                                       align_corners=corner).reshape(self.BS, 1, -1, 1, 3).cuda()
        channels = 24  # 16
        self.offsets = nn.Parameter(torch.randn(2, channels * 2, 3) * 0.05)
        self.layer0 = nn.Conv3d(1, 4, 5, stride=2, bias=False, padding=2)
        self.batch0 = nn.BatchNorm3d(4)

        self.layer1 = nn.Conv3d(channels * 8, channels * 4, 1, bias=False, groups=1)
        self.batch1 = nn.BatchNorm3d(channels * 4)
        self.layer2 = nn.Conv3d(channels * 4, channels * 4, 3, bias=False, padding=1)
        self.batch2 = nn.BatchNorm3d(channels * 4)
        self.layer3 = nn.Conv3d(channels * 4, channels * 1, 1)

    def forward(self, input_img):
        img_in = F.avg_pool3d(input_img, 3, padding=1, stride=2)  # Tensor: (2, 4, 48, 40, 48)
        img_in = F.relu(self.batch0(self.layer0(img_in)))
        sampled = F.grid_sample(img_in, self.ogrid_xyz + self.offsets[0, :, :].reshape(1, -1, 1, 1, 3),
                                align_corners=self.corner).reshape(self.BS, -1, self.o_m, self.o_n, self.o_o)
        sampled -= F.grid_sample(img_in, self.ogrid_xyz + self.offsets[1, :, :].reshape(1, -1, 1, 1, 3),
                                 align_corners=self.corner).reshape(self.BS, -1, self.o_m, self.o_n, self.o_o)
        # Tensor: (2, 192, 64, 53, 64)

        x = F.relu(self.batch1(self.layer1(sampled)))
        x = F.relu(self.batch2(self.layer2(x)))
        features = self.layer3(x)  # Tensor: (2, 24, 64, 53, 64)
        return features


class Seg_Obelisk_Unet(nn.Module):
    def __init__(self, num_labels, full_res):
        super(Seg_Obelisk_Unet, self).__init__()
        D_in2 = full_res[0]  # 192
        H_in2 = full_res[1]  # 160
        W_in2 = full_res[2]  # 192
        D_in3 = D_in2 // 2  # 96
        H_in3 = H_in2 // 2  # 80
        W_in3 = W_in2 // 2  # 96

        D_in4 = D_in3 // 2  # 48
        H_in4 = H_in3 // 2  # 40
        W_in4 = W_in3 // 2  # 48

        D_in5 = D_in4 // 2  # 24
        H_in5 = H_in4 // 2  # 20
        W_in5 = W_in4 // 2  # 24

        D_grid1 = D_in3
        H_grid1 = H_in3
        W_grid1 = W_in3
        D_grid2 = D_in4
        H_grid2 = H_in4
        W_grid2 = W_in4
        D_grid3 = D_in5
        H_grid3 = H_in4
        W_grid3 = W_in5

        self.D_grid1 = D_grid1
        self.H_grid1 = H_grid1
        self.W_grid1 = W_grid1
        self.D_grid2 = D_grid2
        self.H_grid2 = H_grid2
        self.W_grid2 = W_grid2
        self.D_grid3 = D_grid3
        self.H_grid3 = H_grid3
        self.W_grid3 = W_grid3
        # U-Net Encoder
        self.conv0 = nn.Conv3d(1, 4, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(4)
        self.conv1 = nn.Conv3d(4, 16, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(16)
        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch3 = nn.BatchNorm3d(32)

        # Obelisk Encoder
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        self.sample_grid1 = torch.cat((torch.linspace(-1, 1, W_grid2).view(1, 1, -1, 1).repeat(D_grid2, H_grid2, 1, 1),
                                       torch.linspace(-1, 1, H_grid2).view(1, -1, 1, 1).repeat(D_grid2, 1, W_grid2, 1),
                                       torch.linspace(-1, 1, D_grid2).view(-1, 1, 1, 1).repeat(1, H_grid2, W_grid2, 1)),
                                      dim=3).view(1, 1, -1, 1, 3).detach()
        self.sample_grid1.requires_grad = False
        self.sample_grid2 = torch.cat((torch.linspace(-1, 1, W_grid3).view(1, 1, -1, 1).repeat(D_grid3, H_grid3, 1, 1),
                                       torch.linspace(-1, 1, H_grid3).view(1, -1, 1, 1).repeat(D_grid3, 1, W_grid3, 1),
                                       torch.linspace(-1, 1, D_grid3).view(-1, 1, 1, 1).repeat(1, H_grid3, W_grid3, 1)),
                                      dim=3).view(1, 1, -1, 1, 3).detach()
        self.sample_grid2.requires_grad = False
        self.offset1 = nn.Parameter(torch.randn(1, 128, 1, 1, 3) * 0.05)
        self.linear1a = nn.Conv3d(128, 128, 1, groups=2, bias=False)
        self.batch1a = nn.BatchNorm3d(128)
        self.linear1b = nn.Conv3d(128, 32, 1, bias=False)
        self.batch1b = nn.BatchNorm3d(128 + 32)
        self.linear1c = nn.Conv3d(128 + 32, 32, 1, bias=False)
        self.batch1c = nn.BatchNorm3d(128 + 64)
        self.linear1d = nn.Conv3d(128 + 64, 32, 1, bias=False)
        self.batch1d = nn.BatchNorm3d(128 + 96)
        self.linear1e = nn.Conv3d(128 + 96, 16, 1, bias=False)

        self.offset2 = nn.Parameter(torch.randn(1, 512, 1, 1, 3) * 0.05)
        self.linear2a = nn.Conv3d(512, 128, 1, groups=4, bias=False)
        self.batch2a = nn.BatchNorm3d(128)
        self.linear2b = nn.Conv3d(128, 32, 1, bias=False)
        self.batch2b = nn.BatchNorm3d(128 + 32)
        self.linear2c = nn.Conv3d(128 + 32, 32, 1, bias=False)
        self.batch2c = nn.BatchNorm3d(128 + 64)
        self.linear2d = nn.Conv3d(128 + 64, 32, 1, bias=False)
        self.batch2d = nn.BatchNorm3d(128 + 96)
        self.linear2e = nn.Conv3d(128 + 96, 32, 1, bias=False)

        # U-Net Decoder
        self.conv6bU = nn.Conv3d(64, 32, 3, padding=1)  # 96#64#32
        self.batch6bU = nn.BatchNorm3d(32)
        self.conv6U = nn.Conv3d(64, 12, 3, padding=1)  # 64#48
        self.batch6U = nn.BatchNorm3d(12)
        self.conv7U = nn.Conv3d(16, num_labels, 3, padding=1)  # 24#16#24
        self.batch7U = nn.BatchNorm3d(num_labels)
        self.conv77U = nn.Conv3d(num_labels, num_labels, 1)

    def forward(self, inputImg):
        B, C, D, H, W = inputImg.size()
        device = inputImg.device

        leakage = 0.025
        x0 = F.avg_pool3d(inputImg, 3, padding=1, stride=1)
        x00 = F.avg_pool3d(F.avg_pool3d(inputImg, 3, padding=1, stride=1), 3, padding=1, stride=1)

        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), leakage)
        x = F.leaky_relu(self.batch1(self.conv1(x1)), leakage)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)), leakage)
        x = F.leaky_relu(self.batch2(self.conv2(x2)), leakage)
        x = F.leaky_relu(self.batch22(self.conv22(x)), leakage)

        x = F.leaky_relu(self.batch3(self.conv3(x)), leakage)

        x_o1 = F.grid_sample(x0, (self.sample_grid1.to(device).repeat(B, 1, 1, 1, 1) + self.offset1)) \
            .view(B, -1, self.D_grid2, self.H_grid2, self.W_grid2)

        x_o1 = F.leaky_relu(self.linear1a(x_o1), leakage)
        x_o1a = torch.cat((x_o1, F.leaky_relu(self.linear1b(self.batch1a(x_o1)), leakage)), dim=1)
        x_o1b = torch.cat((x_o1a, F.leaky_relu(self.linear1c(self.batch1b(x_o1a)), leakage)), dim=1)
        x_o1c = torch.cat((x_o1b, F.leaky_relu(self.linear1d(self.batch1c(x_o1b)), leakage)), dim=1)
        x_o1d = F.leaky_relu(self.linear1e(self.batch1d(x_o1c)), leakage)
        x_o1 = F.interpolate(x_o1d, size=[self.D_grid1, self.H_grid1, self.W_grid1], mode='trilinear',
                             align_corners=False)

        x_o2 = F.grid_sample(x00, (self.sample_grid2.to(device).repeat(B, 1, 1, 1, 1) + self.offset2)) \
            .view(B, -1, self.D_grid3, self.H_grid3, self.W_grid3)

        x_o2 = F.leaky_relu(self.linear2a(x_o2), leakage)
        x_o2a = torch.cat((x_o2, F.leaky_relu(self.linear2b(self.batch2a(x_o2)), leakage)), dim=1)
        x_o2b = torch.cat((x_o2a, F.leaky_relu(self.linear2c(self.batch2b(x_o2a)), leakage)), dim=1)
        x_o2c = torch.cat((x_o2b, F.leaky_relu(self.linear2d(self.batch2c(x_o2b)), leakage)), dim=1)
        x_o2d = F.leaky_relu(self.linear2e(self.batch2d(x_o2c)), leakage)
        x_o2 = F.interpolate(x_o2d, size=[self.D_grid2, self.H_grid2, self.W_grid2], mode='trilinear',
                             align_corners=False)

        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x, x_o2), 1))), leakage)
        x = F.interpolate(x, size=[self.D_grid1, self.H_grid1, self.W_grid1], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x, x_o1, x2), 1))), leakage)
        x = F.interpolate(x, size=[D, H, W], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch7U(self.conv7U(torch.cat((x, x1), 1))), leakage)

        x = self.conv77U(x)

        return x


class Reg_Obelisk_Unet(nn.Module):
    def __init__(self, full_res):
        super(Reg_Obelisk_Unet, self).__init__()
        D_in2 = full_res[0]  # 192 160
        H_in2 = full_res[1]  # 160 192
        W_in2 = full_res[2]  # 192 160
        D_grid1 = D_in2 // 2  # 96 80
        H_grid1 = H_in2 // 2  # 80 96
        W_grid1 = W_in2 // 2  # 96 80

        D_grid2 = D_grid1 // 2  # 48 40
        H_grid2 = H_grid1 // 2  # 40 48
        W_grid2 = W_grid1 // 2  # 48 40

        D_grid3 = D_grid2 // 2  # 24 20
        H_grid3 = H_grid2  # // 2  # 40 48
        W_grid3 = W_grid2 // 2  # 24 20

        self.D_grid1 = D_grid1
        self.H_grid1 = H_grid1
        self.W_grid1 = W_grid1
        self.D_grid2 = D_grid2
        self.H_grid2 = H_grid2
        self.W_grid2 = W_grid2
        self.D_grid3 = D_grid3
        self.H_grid3 = H_grid2
        self.W_grid3 = W_grid3
        # U-Net Encoder
        self.conv0 = nn.Conv3d(2, 4, 3, padding=1)  # mov and fix will be concatenated in channel dim
        self.batch0 = nn.BatchNorm3d(4)
        self.conv1 = nn.Conv3d(4, 16, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(16)
        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch3 = nn.BatchNorm3d(32)

        # Obelisk Encoder
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        self.sample_grid1 = torch.cat((torch.linspace(-1, 1, W_grid2).view(1, 1, -1, 1).repeat(D_grid2, H_grid2, 1, 1),
                                       torch.linspace(-1, 1, H_grid2).view(1, -1, 1, 1).repeat(D_grid2, 1, W_grid2, 1),
                                       torch.linspace(-1, 1, D_grid2).view(-1, 1, 1, 1).repeat(1, H_grid2, W_grid2, 1)),
                                      dim=3).view(1, 1, -1, 1, 3).detach()
        self.sample_grid1.requires_grad = False
        self.sample_grid2 = torch.cat((torch.linspace(-1, 1, W_grid3).view(1, 1, -1, 1).repeat(D_grid3, H_grid3, 1, 1),
                                       torch.linspace(-1, 1, H_grid3).view(1, -1, 1, 1).repeat(D_grid3, 1, W_grid3, 1),
                                       torch.linspace(-1, 1, D_grid3).view(-1, 1, 1, 1).repeat(1, H_grid3, W_grid3, 1)),
                                      dim=3).view(1, 1, -1, 1, 3).detach()
        self.sample_grid2.requires_grad = False
        self.offset1 = nn.Parameter(torch.randn(1, 128, 1, 1, 3) * 0.05)
        self.linear1a = nn.Conv3d(256, 128, 1, groups=2, bias=False)
        self.batch1a = nn.BatchNorm3d(128)
        self.linear1b = nn.Conv3d(128, 32, 1, bias=False)
        self.batch1b = nn.BatchNorm3d(128 + 32)
        self.linear1c = nn.Conv3d(128 + 32, 32, 1, bias=False)
        self.batch1c = nn.BatchNorm3d(128 + 64)
        self.linear1d = nn.Conv3d(128 + 64, 32, 1, bias=False)
        self.batch1d = nn.BatchNorm3d(128 + 96)
        self.linear1e = nn.Conv3d(128 + 96, 16, 1, bias=False)

        self.offset2 = nn.Parameter(torch.randn(1, 512, 1, 1, 3) * 0.05)
        self.linear2a = nn.Conv3d(1024, 128, 1, groups=4, bias=False)
        self.batch2a = nn.BatchNorm3d(128)
        self.linear2b = nn.Conv3d(128, 32, 1, bias=False)
        self.batch2b = nn.BatchNorm3d(128 + 32)
        self.linear2c = nn.Conv3d(128 + 32, 32, 1, bias=False)
        self.batch2c = nn.BatchNorm3d(128 + 64)
        self.linear2d = nn.Conv3d(128 + 64, 32, 1, bias=False)
        self.batch2d = nn.BatchNorm3d(128 + 96)
        self.linear2e = nn.Conv3d(128 + 96, 32, 1, bias=False)

        # U-Net Decoder
        self.conv6bU = nn.Conv3d(64, 32, 3, padding=1)  # 96#64#32
        self.batch6bU = nn.BatchNorm3d(32)
        self.conv6U = nn.Conv3d(64, 12, 3, padding=1)  # 64#48
        self.batch6U = nn.BatchNorm3d(12)
        self.conv7U = nn.Conv3d(16, 16, 3, padding=1)  # 24#16#24
        self.batch7U = nn.BatchNorm3d(16)

        # for registration, output flow feature get shape: [N, D, W, H, 3]
        self.flow = nn.Conv3d(16, 3, 3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, x):
        # Get encoder activations
        B, C, D, H, W = x.size()
        device = x.device

        leakage = 0.025
        x0 = F.avg_pool3d(x, 3, padding=1, stride=1)
        x00 = F.avg_pool3d(F.avg_pool3d(x, 3, padding=1, stride=1), 3, padding=1, stride=1)

        x1 = F.leaky_relu(self.batch0(self.conv0(x)), leakage)
        x = F.leaky_relu(self.batch1(self.conv1(x1)), leakage)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)), leakage)
        x = F.leaky_relu(self.batch2(self.conv2(x2)), leakage)
        x = F.leaky_relu(self.batch22(self.conv22(x)), leakage)

        x = F.leaky_relu(self.batch3(self.conv3(x)), leakage)

        x_o1 = F.grid_sample(x0, (self.sample_grid1.to(device).repeat(B, 1, 1, 1, 1) + self.offset1)) \
            .view(B, -1, self.D_grid2, self.H_grid2, self.W_grid2)

        x_o1 = F.leaky_relu(self.linear1a(x_o1), leakage)
        x_o1a = torch.cat((x_o1, F.leaky_relu(self.linear1b(self.batch1a(x_o1)), leakage)), dim=1)
        x_o1b = torch.cat((x_o1a, F.leaky_relu(self.linear1c(self.batch1b(x_o1a)), leakage)), dim=1)
        x_o1c = torch.cat((x_o1b, F.leaky_relu(self.linear1d(self.batch1c(x_o1b)), leakage)), dim=1)
        x_o1d = F.leaky_relu(self.linear1e(self.batch1d(x_o1c)), leakage)
        x_o1 = F.interpolate(x_o1d, size=[self.D_grid1, self.H_grid1, self.W_grid1], mode='trilinear',
                             align_corners=False)

        x_o2 = F.grid_sample(x00, (self.sample_grid2.to(device).repeat(B, 1, 1, 1, 1) + self.offset2)) \
            .view(B, -1, self.D_grid3, self.H_grid3, self.W_grid3)

        x_o2 = F.leaky_relu(self.linear2a(x_o2), leakage)
        x_o2a = torch.cat((x_o2, F.leaky_relu(self.linear2b(self.batch2a(x_o2)), leakage)), dim=1)
        x_o2b = torch.cat((x_o2a, F.leaky_relu(self.linear2c(self.batch2b(x_o2a)), leakage)), dim=1)
        x_o2c = torch.cat((x_o2b, F.leaky_relu(self.linear2d(self.batch2c(x_o2b)), leakage)), dim=1)
        x_o2d = F.leaky_relu(self.linear2e(self.batch2d(x_o2c)), leakage)
        x_o2 = F.interpolate(x_o2d, size=[self.D_grid2, self.H_grid2, self.W_grid2], mode='trilinear',
                             align_corners=False)

        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x, x_o2), 1))), leakage)
        x = F.interpolate(x, size=[self.D_grid1, self.H_grid1, self.W_grid1], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x, x_o1, x2), 1))), leakage)
        x = F.interpolate(x, size=[D, H, W], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch7U(self.conv7U(torch.cat((x, x1), 1))), leakage)

        x = self.flow(x)

        return x
