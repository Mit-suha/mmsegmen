# model settings
custom_imports = dict(imports=['mymodels'], allow_failed_imports=False)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)


fuse_dims = [32, 64, 128, 256]
model = dict(
    _scope_='mmseg',
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='HybridBackbone',
        cnn_cfg=dict(
            type="ResNetV1c",
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnet18_v1c',
            ),
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 2, 2),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
        ),
        transformer_cfg=dict(
            type="MixVisionTransformer",
            init_cfg=dict(
                type='Pretrained',
                checkpoint='../pretrained/mit_b0.pth',
            ),
            in_channels=3,
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    ),
    neck=dict(
        type='FuseNeck',
        fuse_types=['ctfuse', 'ctfuse', 'ctfuse', 'ctfuse'],
        residual=True,
        cnn_dims=[64, 128, 256, 512],
        trans_dims=[32, 64, 160, 256],
        fuse_dims=fuse_dims,
    ),
    decode_head=dict(
        type='FPNHead',
        in_channels=fuse_dims,
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
