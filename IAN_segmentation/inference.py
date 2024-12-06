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
import sys
import tempfile
from glob import glob
from time import time
import pandas as pd

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import monai
import json
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.engines import get_devices_spec
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    SaveImage,
    ScaleIntensityd,
    EnsureTyped,
    ScaleIntensityRanged,
    SelectItemsd,
)
from fusion_unet import FusionUNetV3

def infer():
    # monai.config.print_config()
    ROI = (96, 96, 96)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    model_path = os.path.join(dir_name, f"best_2.pth")


    with open(splits_json, 'r') as f:
        files = json.load(f)

    
    test_images = [os.path.join(img_dir, f+".nii.gz") for f in files["test"]]
    test_splines = [os.path.join(spline_dir, f+".nii.gz") for f in files["test"]]

    test_files = [{"img": img, "spline": spline} for img, spline in zip(test_images, test_splines)]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "spline"]),
            EnsureTyped(keys=["img", "spline"]),
            EnsureChannelFirstd(keys=["img", "spline"]),
            # Spacingd(keys=["img", "seg"], pixdim=SPACING),
            ScaleIntensityRanged(keys="img", a_min=-1000, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
            # SelectItemsd(keys=["img", "seg"]),
        ]
    )
    val_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
    # sliding window inference need to input 1 image in every iteration
    val_loader = monai.data.DataLoader(val_ds, batch_size=1)

    post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5, argmax=True)])
    saver = SaveImage(
        output_dir=os.path.join(dir_name, "output_3"),
        output_ext=".nii.gz", 
        output_postfix="", 
        separate_folder=False,
    )
    # try to use all the available GPUs
    devices = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    # devices = get_devices_spec(None)
    model = FusionUNetV3().to(devices[0])

    model.load_state_dict(torch.load(model_path))

    # if we have multiple GPUs, set data parallel to execute sliding window inference
    # if len(devices) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=devices)

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_splines = val_data["img"].to(devices[0]), val_data["spline"].to(devices[0])
            # define sliding window size and batch size for windows inference
            roi_size = ROI
            sw_batch_size = 16
            inputs = torch.cat((val_images, val_splines), dim=1)
            val_outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, overlap=0.5)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            for val_output in val_outputs:
                # np.save(os.path.join(os.path.dirname(test_images[0]), "infer.npy"),val_output)
                saver(val_output)


# Global variables
# data_folder = r'/media/tpx/600C94CE0C94A118/ShuanglinJiang_workspace/ToothFairy_Dataset/Dataset'
# data_folder = r'/media/tpx/600C94CE0C94A118/ShuanglinJiang_workspace/ToothFairy_Dataset/Dataset'
# splits_json = r'D:\Jiangshuanglin\Dataset\splits.json'
# img_dir = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\imagesTr_final'
# seg_dir = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\ian'
# spline_dir = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\spline'

splits_json = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/splits.json'
img_dir = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/images'
seg_dir = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/labels/gt_alpha'
spline_dir = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/splines_test'
abs_path = os.path.abspath(__file__) # absolute path of this file
dir_name = os.path.dirname(abs_path) # directory name of this file
if __name__ == "__main__":
    # groups = [1,] # groups to execute
    # for group in groups: 
    #     print("-"*20)
    #     print(f"Inference group {group}")
    #     infer(group)
    infer()