import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from . NAM import *
from . resnet import *

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]


        self.branch1 = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=1, dilation=1, bias=True, ),
            nn.BatchNorm2d(320, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.nam = Att(24)
        self.nam1 = Att(32)

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):

        low_level_features = self.features[:4](x)

        [b, c, row, col] = low_level_features.size()
        low_level_features1 = self.features[:7](x)
        x = self.features[4:](low_level_features)

        low_level_features1 = self.nam1(low_level_features1)
        low_level_features = self.nam(low_level_features)

        low_level_features1 = F.interpolate( low_level_features1, (row, col), None, 'bilinear', True)

        low_level_features = torch.cat((low_level_features1,low_level_features), dim = 1)

        return low_level_features, x

    # -----------------------------------------#


#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True, ),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )


        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=1 * rate, dilation=1 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=1 * rate, dilation=1 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=1 * rate, dilation=1 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch5_pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.branch5_pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.branch5_conv0 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_in, momentum=bn_mom)
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.nam = Att(256)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(conv3x3_1)
        conv3x3_3 = self.branch4(conv3x3_2)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#

        x1 = F.interpolate(self.branch5_pool1(x), (row, col), None, 'bilinear', True)
        x2 = F.interpolate(self.branch5_pool2(x), (row, col), None, 'bilinear', True)
        x1 = self.branch5_conv0(x1)
        x2 = self.branch5_conv0(x2)
        x1 = F.relu_(x + x1)
        x2 = F.relu_(x + x2)
        global_feature = F.relu_(x1+x2)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)

        result = self.conv_cat(feature_cat)
        result = self.nam(result)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 56
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        self.resnet = Bottleneck(48,12)


    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)


        low_level_features = self.shortcut_conv(low_level_features)

        low_level_features = self.resnet(low_level_features)
        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

