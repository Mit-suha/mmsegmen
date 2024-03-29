from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset



import os.path as osp
import numpy as np
import mmcv
from PIL import Image


@DATASETS.register_module()
class HrcDataset(BaseSegDataset):

    CLASSES = ("background", "cloud")

    PALETTE = [
        [0, 0, 0],
        [255, 255, 255],
    ]

    def __init__(self, **kwargs):
        super(HrcDataset, self).__init__(
            img_suffix=".tif", seg_map_suffix=".tif", reduce_zero_label=False, **kwargs
        )
        assert osp.exists(self.img_dir)