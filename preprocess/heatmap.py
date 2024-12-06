import scipy.ndimage as ndi
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def Mask2Heatmap():
    '''Convert binary mask to heatmap'''
    input_folder = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/nerveSlices'
    output_folder = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/heatmaps'
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # files = ['1000995722_20180112_087.png']
    files = tqdm(files)
    for file in files:
        files.set_description(file)
        file_path = os.path.join(input_folder, file)
        img = np.array(Image.open(file_path).convert('L'))
        kernel = np.ones((3, 3), dtype=np.uint8)
        # Closing operation
        closing = ndi.binary_dilation(img, kernel, iterations=1)
        # Distance transform
        dist_transform = np.sqrt(ndi.distance_transform_edt(closing)**2)
        # Normalize distance transform
        dist_transform = (dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255
        # Store heatmap
        heatmap = Image.fromarray(dist_transform.astype(np.uint8))
        heatmap.save(os.path.join(output_folder, file))

Mask2Heatmap()