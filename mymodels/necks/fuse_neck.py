# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.model.weight_init import(constant_init, normal_init,
                                        trunc_normal_init)


from mmseg.models.builder import NECKS


class CTFUSE(nn.Module):

    def __init__(self,
                 channels_trans,
                 channels_cnn,
                 channels_fuse,
                 residual=False):
        super(CTFUSE, self).__init__()
        self.residual = residual
        self.conv_trans = nn.Conv2d(
            channels_trans, channels_fuse, kernel_size=1)
        self.conv_cnn = nn.Conv2d(channels_cnn, channels_fuse, kernel_size=1)
        self.conv_fuse = nn.Conv2d(
            2 * channels_fuse, 2 * channels_fuse, kernel_size=1)
        self.linear_fuse = nn.Sequential(
            nn.BatchNorm2d(2 * channels_fuse, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(2 * channels_fuse, channels_fuse, kernel_size=1),
        )
        if self.residual:
            self.conv_residual = nn.Conv2d(
                channels_trans + channels_cnn, channels_fuse, kernel_size=1)

    def forward(self, trans_input, cnn_input):
        """x:transformer, y:cnn"""

        if self.residual:
            residual = self.conv_residual(
                torch.cat((trans_input, cnn_input), dim=1))
        trans_input = self.conv_trans(trans_input)
        cnn_input = self.conv_cnn(cnn_input)
        fuse = self.conv_fuse(torch.cat((trans_input, cnn_input), dim=1))
        fuse = self.linear_fuse(fuse)
        if self.residual:
            fuse = fuse + residual
        return fuse


class Plain(nn.Module):

    def __init__(self, channels_trans, channels_cnn, channels_fuse):
        super(Plain, self).__init__()
        self.conv_trans = nn.Conv2d(
            channels_trans, channels_fuse, kernel_size=1)
        self.conv_cnn = nn.Conv2d(channels_cnn, channels_fuse, kernel_size=1)

    def forward(self, x, y):
        x = self.conv_trans(x)
        y = self.conv_cnn(y)
        return x + y


@NECKS.register_module()
class FuseNeck(BaseModule):

    def __init__(
            self,
            fuse_types,
            fuse_dims,
            residual=True,
            cnn_dims=[64, 128, 256, 512],
            trans_dims=[32, 64, 160, 256],
            init_cfg=None
    ):
        super(FuseNeck, self).__init__(init_cfg=init_cfg)

        self.fuse_types = fuse_types
        self.num_inputs = len(trans_dims)
        self.fuse_dims = fuse_dims
        self.trans_dims = trans_dims
        self.cnn_dims = cnn_dims
        self.residual = residual
        self.fuse_modules = self._build_fuse_modules()

    def _build_fuse_modules(self):
        assert self.fuse_types != None
        fuse_modules = nn.ModuleList()
        for i in range(self.num_inputs):
            type = self.fuse_types[i]
            assert type in ["lafm", "cafm", "ctfuse", "plain"]
            if type == "ctfuse":
                fuse_modules.append(
                    CTFUSE(
                        channels_trans=self.trans_dims[i],
                        channels_cnn=self.cnn_dims[i],
                        channels_fuse=self.fuse_dims[i],
                        residual=self.residual,
                    ))
            elif type == 'plain':
                fuse_modules.append(
                    Plain(
                        channels_trans=self.trans_dims[i],
                        channels_cnn=self.cnn_dims[i],
                        channels_fuse=self.fuse_dims[i],
                    ))
        return fuse_modules

    def forward(self, inputs):
        trans_inputs, cnn_inputs = inputs
        outs = []
        for i in range(self.num_inputs):
            trans_input = trans_inputs[i]
            cnn_input = cnn_inputs[i]
            feat_fuse = self.fuse_modules[i](
                trans_input=trans_input, cnn_input=cnn_input)
            outs.append(feat_fuse)
        return outs

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)