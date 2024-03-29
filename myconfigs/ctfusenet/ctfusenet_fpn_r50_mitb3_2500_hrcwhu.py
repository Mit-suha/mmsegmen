_base_ = [
    './ctfusenet_fpn_r18_mitb0_2500_hrcwhu.py',
]


fuse_dims = [64, 128, 256, 512]

crop_size = (640, 360)

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        cnn_cfg=dict(
            depth=50,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnet50_v1c',
            ),
        ),
        transformer_cfg=dict(
            init_cfg=dict(
                type='Pretrained',
                checkpoint='../pretrained/mit_b3.pth',
            ),
            embed_dims=64,
            num_layers=[3, 4, 18, 3],
        ),
    ),
    neck=dict(
        cnn_dims=[256, 512, 1024, 2048],
        trans_dims=[64, 128, 320, 512],
        fuse_dims=fuse_dims,
    ),
    decode_head=dict(
        in_channels=fuse_dims,
        channels=256,
    ),
)