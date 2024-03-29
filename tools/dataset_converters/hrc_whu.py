# 复制图像到另一个文件夹
import os
import os.path as osp
from os import mkdir
import shutil
import random
# 创建一个子文件存放文件
out_dir = 'H:\\openmmlab\\HRC_WHU'
mkdir(out_dir)
mkdir(osp.join(out_dir, 'images'))
mkdir(osp.join(out_dir, 'images', 'training'))
mkdir(osp.join(out_dir, 'images', 'validation'))
mkdir(osp.join(out_dir, 'images', 'test'))
mkdir(osp.join(out_dir, 'annotations'))
mkdir(osp.join(out_dir, 'annotations', 'training'))
mkdir(osp.join(out_dir, 'annotations', 'validation'))
mkdir(osp.join(out_dir, 'annotations', 'test'))

train_dir = osp.join(out_dir, 'images', 'training')
val_dir = osp.join(out_dir, 'images', 'validation')
test_dir = osp.join(out_dir, 'images', 'test')

train_ann_dir = osp.join(out_dir, 'annotations', 'training')
val_ann_dir = osp.join(out_dir, 'annotations', 'validation')
test_ann_dir = osp.join(out_dir, 'annotations', 'test')

# 文件所在文件根目录
root_dir = 'H:\\openmmlab\\HRC_WHU0'
image_dir = os.path.join(root_dir, "image")
image_list = os.listdir(image_dir)
total_sample = len(image_list)

ann_dir = os.path.join(root_dir, "label")
ann_list = os.listdir(ann_dir)




for i in range(int(0.3*total_sample)):
    randi = random.randint(0,total_sample-1-i)
    print(image_list[randi])
    print(ann_list[randi])
    shutil.copy(os.path.join(image_dir, image_list[randi]), val_dir)
    shutil.copy(os.path.join(image_dir, image_list[randi]), test_dir)

    shutil.copy(os.path.join(ann_dir, ann_list[randi]), val_ann_dir)
    shutil.copy(os.path.join(ann_dir, ann_list[randi]), test_ann_dir)

    ann_list.remove(ann_list[randi])
    image_list.remove(image_list[randi])

for i in range(len(image_list)):
    shutil.copy(os.path.join(image_dir, image_list[i]), train_dir)


    shutil.copy(os.path.join(ann_dir, ann_list[i]), train_ann_dir)


# for image in file_list:
#
#     # 如果图像名为B.png 则将B.png复制到F:\\Test\\TestA\\class
#     if image == "B.png":
#         if os.path.exists(os.path.join(file_dir, 'class_name')):
#             shutil.copy(os.path.join(file_dir, image), os.path.join(file_dir, 'class_name'))
#         else:
#             os.makedirs(os.path.join(file_dir, 'class_name'))
#             shutil.copy(os.path.join(file_dir, image), os.path.join(file_dir, 'class_name'))

