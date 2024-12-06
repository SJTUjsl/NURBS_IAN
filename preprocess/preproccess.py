import SimpleITK as sitk
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
from geomdl import fitting
from skimage import morphology
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from multiprocessing import Pool

def ApproxLabel(label):
    '''将下颌神经粗分割结果拟合为两条样条曲线，使用时直接调用该函数即可'''
    # label: 三维二值图像
    # Return: 拟合后的样条曲线label

    # 1.将右侧下颌神经标签设为2，以方便对两根神经分开拟合
    split_line = label.shape[2] // 2
    mask = (label == 1) & (np.arange(label.shape[2]) >= split_line)
    label[mask] = 2
    # 左
    label_l = label.copy()
    mask_l = (label_l==2)
    label_l[mask_l] = 0
    # 右
    label_r = label.copy()
    mask_r = (label_r==1)
    label_r[mask_r] = 0

    # 2. 拟合样条曲线
    curve_l = ApproxOneCurve(label_l)
    curve_r = ApproxOneCurve(label_r)
    curve = curve_l + curve_r # 将左右两条曲线叠加为一张图像

    # 3. 返回拟合曲线
    curve = curve.astype(np.uint8)
    return curve

def ApproxOneCurve(label):
    '''将散点拟合为一条样条曲线'''
    # label: 三维二值图像
    # Return: 拟合后的样条曲线label
    if label.sum() == 0: # 如果没有有效点，直接返回空的label
        return label
    
    # 1. 提取骨架
    # morphology.remove_small_objects(label.astype(bool), min_size=500, out=label)
    label = keep_largest_component(label.astype(bool))
    skeleton = morphology.skeletonize(label.astype(np.uint8))

    # 2. 将骨架转化为坐标表示：x, y, z
    x, y , z = np.where(skeleton > 0)
    coordinates = np.column_stack((x, y, z))
    coordinates_list = list(np.array(coordinates))

    # 3. 重新排列样本点
    # 使用深度优先搜索对样本点进行排序
    # _, start_index = SortPointsByDFS(coordinates_list) # 第一次遍历，找到其中一个端点
    # sorted_coordinates, _ = SortPointsByDFS(coordinates_list, start_index) # 第二次遍历，从第一个端点开始遍历

    # 3. 使用角度排序
    # 计算参考点
    points_max = np.max(coordinates_list, axis=0)
    points_min = np.min(coordinates_list, axis=0)
    reference_point = np.array(
        [points_max[0], (points_min[1] + points_max[1]) // 2, (points_min[2] + points_max[2]) // 2])
    # 排序
    sorted_coordinates = SortPointsByAngle(coordinates_list, reference_point)
    sorted_coordinates_list = list(np.array(sorted_coordinates))

    # 4. 拟合样条曲线，得到一批曲线样本点evalpts
    curve = fitting.approximate_curve(sorted_coordinates_list, degree=3, ctrlpts_size=7)
    curve.delta = 0.005
    evalpts = np.array(curve.evalpts)

    # 5. 通过膨胀操作将曲线膨胀为管柱
    tube_volume = np.zeros(label.shape) # 新建一个空的tube_array
    evalpts = evalpts.astype(int)# 将拟合曲线的坐标点转化为整数
    tube_volume[evalpts[:,0], evalpts[:,1], evalpts[:,2]] = 1# 将拟合曲线的坐标点赋值为1
    dilated_volume = morphology.dilation(tube_volume, morphology.cube(3)) # 膨胀操作，膨胀三个像素单位

    return dilated_volume

def keep_largest_component(binary_image):
    # Label the connected components
    labeled_image, num_features = ndimage.label(binary_image)
    # Find the size of each component
    sizes = ndimage.sum(binary_image, labeled_image, range(num_features + 1))
    # Identify the largest component (excluding the background)
    largest_component_label = sizes[1:].argmax() + 1
    # Create a new binary image where only the largest component is kept
    largest_component = (labeled_image == largest_component_label)
    return largest_component

def SortPointsByAngle(points, reference_point):
    '''按照点到参考点的连线与x轴的夹角对点集进行排序'''
    # points: list of points
    # return: sorted points
    points = np.array(points)
    # 计算点到参考点的连线与x轴的夹角
    angles = []
    for point in points:
        vector = point - reference_point
        angle = np.arctan2(vector[1], vector[0]) # 返回值为[-pi, pi]
        angles.append(angle)
    angles = np.array(angles)
    angles = (angles + 2*np.pi) % (2*np.pi) # 将角度转化到[0, 2pi]区间
    # 按照角度排序
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def main(label_name):
    # 文件路径
    if label_name in os.listdir(spline_dir):
        return
    label_path = os.path.join(label_dir, label_name)
    label_image = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label_image)

    # 拟合下颌神经
    curve = ApproxLabel(label_array)
    # 保存拟合结果
    curve_image = sitk.GetImageFromArray(curve)
    curve_image.CopyInformation(label_image)
    sitk.WriteImage(curve_image, os.path.join(spline_dir, label_name))

    # 可视化
    z,x,y = label_array.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c= 'gray', alpha=0.05)
    z,x,y = curve.nonzero()
    ax.scatter(x, y, z, zdir='z', c= 'red', alpha=1)
    plt.savefig(os.path.join(r'/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/splines/snapshot', label_name.split('.')[0] + '.png'))
    plt.close()

# Global varaibles
label_dir = r'/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/labels/gt_alpha'
spline_dir = r'/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/splines'

os.makedirs(spline_dir, exist_ok=True)
if __name__ == '__main__':
    label_list = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]
    with Pool(4) as p:
        p.map(main, label_list)


    # label_list = ['MandibularNerve033.nii.gz', 'MandibularNerve035.nii.gz']
    # label_names = tqdm(label_list)
    # for label_name in label_names:
    #     # if label_name in os.listdir(spline_dir):
    #     #     continue
    #     label_names.set_description('Processing %s' % label_name)
    #     label_path = os.path.join(label_dir, label_name)
    #     label_image = sitk.ReadImage(label_path)
    #     label_array = sitk.GetArrayFromImage(label_image)

    #     # 拟合下颌神经
    #     curve = ApproxLabel(label_array)
    #     # 保存拟合结果
    #     curve_image = sitk.GetImageFromArray(curve)
    #     curve_image.CopyInformation(label_image)
    #     sitk.WriteImage(curve_image, os.path.join(spline_dir, label_name))

    #     # 可视化
    #     z,x,y = label_array.nonzero()
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(x, y, z, zdir='z', c= 'gray', alpha=0.05)
    #     z,x,y = curve.nonzero()
    #     ax.scatter(x, y, z, zdir='z', c= 'red', alpha=1)
    #     plt.savefig(os.path.join(r'D:\Work Space\SegMandibularNerve\Dataset\nerveSpline_GT\snapshot', label_name.split('.')[0] + '.png'))
    #     plt.close()


#############################Single Test##################################
    # 文件路径
    # label_path = r'E:\Dataset\labels\MandibularNerve\1001172283_20190622.nii.gz'
    # label_image = sitk.ReadImage(label_path)
    # label_array = sitk.GetArrayFromImage(label_image)

    # # 拟合下颌神经
    # curve = ApproxLabel(label_array)
    # # 保存拟合结果
    # curve_image = sitk.GetImageFromArray(curve)
    # curve_image.CopyInformation(label_image)
    # # sitk.WriteImage(curve_image, r'E:\Dataset\labels\nerveSpline\1001172283_20190622.nii.gz')

    # # 可视化
    # z,x,y = label_array.nonzero()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, zdir='z', c= 'gray', alpha=0.05)
    # z,x,y = curve.nonzero()
    # ax.scatter(x, y, z, zdir='z', c= 'red', alpha=1)
    # # plt.savefig(r'E:\Dataset\labels\nerveSpline\snapshot\1001172283_20190622.png')
    # plt.show()