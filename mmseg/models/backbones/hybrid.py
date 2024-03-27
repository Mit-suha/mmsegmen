# Copyright (c) OpenMMLab-yue. All rights reserved.
from mmengine.model import BaseModule
from ..builder import BACKBONES


@BACKBONES.register_module()
class HybridBackbone(BaseModule):

    def __init__(self, cnn_cfg, transformer_cfg, init_cfg=None):
        super(HybridBackbone, self).__init__(init_cfg=init_cfg)

        self.transformer = BACKBONES.build(transformer_cfg)
        self.cnn = BACKBONES.build(cnn_cfg)

    def forward(self, x):
        transformer_outs = self.transformer(x)
        cnn_outs = self.cnn(x)
        return transformer_outs, cnn_outs