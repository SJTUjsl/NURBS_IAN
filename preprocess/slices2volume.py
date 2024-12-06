import SimpleITK as sitk
import numpy as np
import os
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

def slice2volume(dir, filename, info):
    '''Restore the volume from slices'''
    """
    Args:
        dir: the folder of slices
        filename: the basename of slices
        info: list of dictionary, containing shape, origin, spacing, direction and transformation matrix of each slice for one volume
    Returns:
        volume: the restored itk volume
    """
    # get all slices
    # print(filename)
    # print(info)
    slices = [f for f in os.listdir(dir) if f.startswith(filename+'_') and f.endswith('.png')]
    # create an empty volume
    volume = np.zeros(info['shape'], dtype=np.uint8)
    # restore the volume from slices
    for slice in slices: 
        # get the index of slice
        index = slice.split('_')[-1].split('.')[0]
        transform = np.array(info[index]) # get the transformation matrix
        # print("transform: ", transform)
        # load slice
        im = Image.open(os.path.join(dir, slice))
        slice = np.array((im), dtype=np.uint8)
        row, col = np.where(slice > 0) # get the non-zero pixels
        # print(row, col)
        row -= slice.shape[0] // 2 # shift the coordinates
        coordiantes = np.stack((np.zeros_like(row), row, col, np.ones_like(row)), axis=-1) # get the coordinates of non-zero pixels
        # print(coordiantes)

        # calculate the coordinates of pixels in volume
        vol_coordinates = np.floor(np.squeeze(transform @ coordiantes[..., None], axis=-1)).astype(int)

        # np.floor(vol_coordinates, out=vol_coordinates).astype(np.int16, copy=False) # round the coordinates to integers
        # print(index, vol_coordinates.shape)
        try:
            volume[vol_coordinates[:, 0], vol_coordinates[:, 1], vol_coordinates[:, 2]] = 1
        except Exception as e:
            print(f'filename:{filename}, slice index:{index}, error:{e}')
            pass
        # print(np.max(slice), np.sum(slice), np.sum(volume))
    # convert the numpy array to itk volume
    # plt.imshow(volume[50,:,:], cmap='gray')
    volume = sitk.GetImageFromArray(volume)
    volume.SetSpacing(tuple(info['spacing']))
    volume.SetOrigin(tuple(info['origin']))
    volume.SetDirection(tuple(info['direction']))
    return volume



slices_dir = r'/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/nerveSlices'
json_dir = r'/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/info.json'
out_dir = r'/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/infer_3D'
with open(json_dir, 'r') as f:
    info_dict = json.load(f)
# filenames = list(info_dict.keys())
with open("/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/splits.json", 'r') as f:
    splits = json.load(f)
test = splits["test"]
filenames = [f+'.nii.gz' for f in test]
print(filenames)
# Single test
# filename = 'MandibularNerve061.nii.gz'
# info = info_dict[filename]
# volume = slice2volume(slices_dir, filename.split('.')[0], info)
# sitk.WriteImage(volume, os.path.join(out_dir, filename.split('.')[0] + '.nii.gz'))
####################################################

# run on all 
filenames = tqdm(filenames)
for filename in filenames:
    filenames.set_description(filename)
    info = info_dict[filename]
    volume = slice2volume(slices_dir, filename.split('.')[0], info)
    sitk.WriteImage(volume, os.path.join(out_dir, filename.split('.')[0] + '.nii.gz'))