_base_ = [
    '../_base_/models/ctfusenet_r18_mitb0.py',
    '../_base_/datasets/hrc_whu.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_2500.py',
]


fuse_dims = [64, 128, 256, 512]

crop_size = (640, 360)

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    neck=dict(
        fuse_dims=fuse_dims,
    ),
    decode_head=dict(in_channels=fuse_dims, ),
)