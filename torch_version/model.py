import copy
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchstat import stat


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, bias=False, **kwargs):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNReLU_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNReLU_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DWConvBNReLU, self).__init__()
        padding = padding if padding else int(kernel_size/2)
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, norm_layer=norm_layer),
            ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.conv(x)


class UpSampleConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, stride=1):
        super(UpSampleConvReLU, self).__init__()
        self.scale_factor = scale_factor
        self.conv = ConvReLU(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.conv(x)
        return x


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()
        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)
        ))

        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)
        return out


class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = UpSampleConvReLU(128, 128)
        self.dw1 = DWConvBNReLU(128, 128)
        self.up2 = UpSampleConvReLU(128, 128)
        self.dw2 = DWConvBNReLU(128, 128)

    def forward(self, x):
        f1, f2, f3 = x
        x = self.up1(f3) + f2
        x = self.dw1(x)

        x = self.up2(x) + f1
        x = self.dw2(x)
        return x



class MSTCN_LPR(nn.Module):
    def __init__(self, target_size=(32,96), num_chars=85):
        super(MSTCN_LPR, self).__init__()
        self.conv0 = ConvBNReLU(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        self.dsconv1 = DWConvBNReLU(in_channels=64, out_channels=128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.dsconv2 = DWConvBNReLU(in_channels=128, out_channels=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dsconv3 = DWConvBNReLU(in_channels=128, out_channels=128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.decoder = Decoder()
        self.upbottom = UpSampleConvReLU(128, 128, 4)
        # self.tcn = MS_TCN2(4, 4, )
        self.output = nn.Sequential(
            nn.Conv2d(128, num_chars, [8, 1], stride=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv0(x)
        x = self.maxpool0(x)

        x = self.dsconv1(x)
        f1, _ = torch.split(x, 8, dim=2)
        x = self.maxpool1(x)

        x = self.dsconv2(x)
        f2, _ = torch.split(x, 4, dim=2)
        x = self.maxpool2(x)

        x = self.dsconv3(x)
        f3, bottom = torch.split(x, 2, dim=2)
        bottom = self.upbottom(bottom)

        x = self.decoder([f1, f2, f3])
        x = torch.concat((x, bottom), dim=-1)
        x = self.output(x).squeeze(2).permute(0, 2, 1)
        return x


if __name__ == '__main__':
    model = MSTCN_LPR()

    target_size = (32, 96)
    h, w = target_size
    test_data = torch.rand((1, 1, h, w))
    out = model(test_data)
    print(out.shape)
    quit()
    stat(model, (1, h, w))

    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, (1, h, w),
            as_strings=True, print_per_layer_stat=True, verbose=True
        )
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
