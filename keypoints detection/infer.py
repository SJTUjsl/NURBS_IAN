# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import sys
from glob import glob

import torch
from PIL import Image
import torch.nn.functional as F
import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Compose,
    LoadImaged,
    SaveImage,
    ScaleIntensityd,
)


def main():
    # monai.config.print_config()
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    images_folder = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\slices\imageSlices'
    images = sorted(glob(os.path.join(images_folder, "*.png")))
    val_files = [{"img": img} for img in images]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # sliding window inference need to input 1 image in every iteration
    val_loader = DataLoader(val_ds, batch_size=36, num_workers=24, collate_fn=list_data_collate)
    post_trans = Compose([Activations(other=lambda x: F.relu(x)), AsDiscrete(threshold=0.98)])
    # post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold_values=0.5), KeepLargestConnectedComponent()])
    saver = SaveImage(output_dir=r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\slices\imageSlices\infer', output_ext=".png", output_postfix="", separate_folder=False, scale=255)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2, 
        in_channels=1, 
        out_channels=1,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2), 
        num_res_units=2,
    ).to(device)
    
    def remove_module_prefix(state_dict):
        # 新建一个空字典用于存储更新后的状态字典
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除键名中的'module.'前缀
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    # 加载模型参数
    state_dict = torch.load("toothfairy.pth")

    # 移除'module.'前缀
    updated_state_dict = remove_module_prefix(state_dict)

    # 加载更新后的状态字典到模型中
    model.load_state_dict(updated_state_dict)

    model.eval()
    iou = MeanIoU(include_background=False, reduction="mean", ignore_empty=False)
    dice = DiceMetric(include_background=False, reduction="mean", ignore_empty=False)
    iou_metrics = []
    dice_metrics = []
    with torch.no_grad():
        for val_data in val_loader:
            val_images = val_data["img"].to(device)
            # define sliding window size and batch size for windows inference
            # roi_size = (224, 224)
            # sw_batch_size = 16
            val_outputs = model(val_images)
            # val_outputs = post_trans(val_outputs)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            iou(val_outputs, val_images)
            dice(val_outputs, val_images)
            iou_metrics.append(iou.aggregate().item())
            dice_metrics.append(dice.aggregate().item())
            for val_output in val_outputs:
                saver(val_output)
    print("Mean IoU:", np.mean(iou_metrics))
    print("Mean Dice:", np.mean(dice_metrics))

def evaluate():
    pred_folder = r'./infer'
    gt_folder = r'./labels'
    import seg_metrics.seg_metrics as sg
    csv_file = './metric.csv'

    filenames = os.listdir(pred_folder)
    pred_paths = [os.path.join(pred_folder, f) for f in filenames]
    gt_paths = [os.path.join(gt_folder, f) for f in filenames]
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred = Image.open(pred_path).convert('L')
        pred = np.array(pred)
        pred[pred > 0] = 1
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt[gt > 0] = 1
        res = sg.write_metrics(
            labels=[1,],
            gdth_img=gt,
            pred_img=pred,
            csv_file=csv_file,
        )
        print(res)

if __name__ == "__main__":
    main()
    # evaluate()

