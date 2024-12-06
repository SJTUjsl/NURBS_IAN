import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy.optimize import root_scalar
import os
import json
# from geomdl import fitting
from tqdm import tqdm
import skimage
from skimage.morphology import dilation, cube, skeletonize
from PIL import Image


# def get_coordinates(arr):
#     """Get coordinate of points with value 1"""
#     rows, cols = np.where(arr == 1)
#     coordinates = np.column_stack((cols, rows))
#     return coordinates


def normal2base(normal):
    """根据平面法向量求三维坐标系的另外两个基向量"""
    if not isinstance(normal, np.ndarray):
        normal = np.array(normal)
    # Calculate the first base vector
    base2 = np.array([1, 0, 0])
    base1 = np.cross(base2, normal)
    # Normalize the base vectors
    base1 = base1 / np.linalg.norm(base1)
    base2 = base2 / np.linalg.norm(base2)
    return base1, base2


def schmidt(base1, base2):
    '''施密特正交单位化'''
    base1 = np.array(base1)
    base2 = np.array(base2)
    base1 = base1 / np.linalg.norm(base1)
    base2 = base2 / np.linalg.norm(base2)
    base3 = np.cross(base1, base2)
    base3 = base3 / np.linalg.norm(base3)
    return base3

def resample(x_lim, polyfunc: np.poly1d, num_pts):
    """Resample a curve defined by a polynomial function to get evenly spaced points"""
    """
    Args:
        x_lim: 采样的x坐标范围
        polyfunc: 多项式函数
        num_pts: 采样点数
    """
    x1, x2 = x_lim
    my_poly = polyfunc
    num_points = num_pts
    def curve_length(x):
        return spi.quad(lambda t: np.sqrt(1 + my_poly(t)**2), x1, x)[0] # 求定积分
    total_curve_length = curve_length(x2)

    # 计算等间距点之间的理论距离
    desired_spacing = total_curve_length / (num_points - 1)

    # 定义函数来查找使得两点间距离等于 desired_spacing 的 x 坐标
    def find_x_for_distance(d):
        result = root_scalar(lambda x: curve_length(x) - d, bracket=[x1, x2], method='bisect')
        return result.root

    # 生成均匀间隔的点
    interpolated_x = [x1]
    for i in range(1, num_points - 1):
        desired_length = i * desired_spacing
        interpolated_x.append(find_x_for_distance(desired_length))
    interpolated_x.append(x2)

    # 计算每个点对应的 y 值
    interpolated_y = [my_poly(x) for x in interpolated_x]

    # 打印插值点
    interpolated_points = np.column_stack((interpolated_x, interpolated_y))

    return interpolated_points



def extract_slices_new(image_path, mandible_path, nerve_path, num_pieces=40, train=False):
    '''提取一个CT中的所有二维切片'''
    """
    Args:
        image_path: 原始图像路径
        mandible_path: 下颌骨mask路径
        nerve_path: 神经label路径
        num_pieces: 提取切片的数量
        train: 是否是用于训练的前处理
    Returns:
        image_slices_list: CT切片数组
        nerve_slices_list: 神经切片数组
    """
    
    slice_size = (160, 160) # 切片大小
    image_itk = sitk.ReadImage(image_path)
    mandible_itk = sitk.ReadImage(mandible_path)
    image_itk = sitk.Cast(sitk.IntensityWindowing(
        image_itk,windowMinimum=-1000,windowMaximum=1500),sitk.sitkUInt8) # 截断灰度范围并作归一化
    image_array = sitk.GetArrayFromImage(image_itk)
    mandible_array = sitk.GetArrayFromImage(mandible_itk)
    if train and nerve_path is not None:
        nerve_itk = sitk.ReadImage(nerve_path)
        nerve_array = sitk.GetArrayFromImage(nerve_itk)
        nerve_array = dilation(nerve_array, cube(3)) # 将神经label适当膨胀，增加像素点
    else:
        nerve_array = None

    dilated_roi = dilation(mandible_array, cube(21))  # 截取下颌骨区域，并适当膨胀作为ROI
    image_array[dilated_roi == 0] = 0

    '''1.找拟合平面'''
    ### 方案一：直接用label点最多的层作为arch层
    # num_label = np.sum(mandible_array, axis=(1, 2)) # 像素点最多的平面
    # arch_index = np.argmax(num_label) - len(num_label) // 20
    ### 方案二：用下颌骨区域下25%的层作为arch层，将整个下颌骨纵向展平用于拟合曲线
    # bbox = np.where(mandible_array != 0)
    # min_z = np.min(bbox[0])
    # max_z = np.max(bbox[0])
    # arch_index = min_z + (max_z - min_z) // 4

    '''2.提取骨架并拟合出牙弓曲线'''
    flatten_image = np.sum(mandible_array, axis=0) > 0 # 将下颌骨纵向展平
    flatten_image = skimage.morphology.opening(flatten_image, skimage.morphology.square(19))
    image_skeleton = skeletonize(flatten_image.astype(np.uint8)) # 提取骨架
    rows, cols = np.where(image_skeleton == 1) # 骨架坐标
    skeleton_points = np.column_stack((cols, rows)) # 骨架点列表
    # 骨架点排序
    skeleton_points = np.array(sorted(skeleton_points, key=lambda x: x[0])) # 按x坐标排序

    # 等间隔取点用于拟合牙弓曲线
    # control_points_inx = np.linspace(0, len(skeleton_points)-1, 40, dtype=int)
    # control_points = skeleton_points[control_points_inx, :] # 控制点坐标
    # 用二次多项式进行拟合
    eff = np.polyfit(skeleton_points[:, 0], skeleton_points[:, 1], 2)
    p = np.poly1d(eff) # 牙弓曲线表达式

    '''3.计算采样点的坐标和切片的法向量'''
    # 采样点坐标
    x_lim = [skeleton_points[:, 0].min(), skeleton_points[:,0].max()]
    x = np.linspace(x_lim[0], x_lim[1], num_pieces)
    # y = p(x)
    # x_lim = [skeleton_points[0][0], skeleton_points[-1][0]]
    centers = resample(x_lim, p, num_pieces)  # 重采样
    # centers = np.column_stack((x, y)) # 采样点坐标
    # 采样点切向量
    p_deriv = np.polyder(p)
    normals = np.column_stack((np.ones_like(x), p_deriv(x))) # 法向量
    normals = normals / np.linalg.norm(normals, axis=1)[:, None] # Normalization

    '''4.获取切片'''
    # 切片数组
    image_slice_list = []
    nerve_slice_list = []
    # 坐标变换矩阵
    A_aug_dict = {}
    A_aug_dict["shape"] = image_array.shape
    A_aug_dict["origin"] = image_itk.GetOrigin()
    A_aug_dict["spacing"] = image_itk.GetSpacing()
    A_aug_dict["direction"] = image_itk.GetDirection()
    counter = 0 # 计数器
    mandible_min = np.argwhere(mandible_array == 1)[0][0] # 下颌骨最低点
    for center, normal2d in zip(centers, normals): # 遍历采样点, 计算每个点对应的坐标变换矩阵
        # 创建切片
        image_slice = np.zeros(slice_size)
        label_slice = np.zeros(slice_size)
        # 切片二维坐标
        i, j = np.indices(image_slice.shape)
        i = i - slice_size[0] // 2  # 坐标原点不在切片的角落，而是在正下方中心，在还原的时候要注意对横坐标进行平移
        # j = j + mandible_min
        # 计算平移向量
        translation = np.array([[mandible_min], [center[1]], [center[0]]])
        # 计算旋转矩阵
        normal = [0, normal2d[1], normal2d[0]]
        base2 = np.array([1, 0, 0])
        base1 = schmidt(normal, base2) # 三个基向量
        A = np.array([normal, base1, base2]).T # 旋转矩阵
        # 坐标转换矩阵
        A_aug = np.hstack((A, translation))
        A_aug = np.vstack((A_aug, np.array([0, 0, 0, 1])))
        
        # 切片三维增广坐标
        x_aug = np.stack([np.zeros_like(i), i, j, np.ones_like(i)], axis=-1)
        # 计算切片中各点在原坐标系下的坐标
        x_aug_original = np.squeeze(A_aug @ x_aug[..., None]).astype(int)
        # 保留在原图像范围内的点
        valid = np.logical_and.reduce((x_aug_original[..., 0] >= 0, x_aug_original[..., 0] < image_array.shape[0],
                                       x_aug_original[..., 1] >= 0, x_aug_original[..., 1] < image_array.shape[1],
                                       x_aug_original[..., 2] >= 0, x_aug_original[..., 2] < image_array.shape[2]))
        
        if nerve_array is not None: # 训练模式
            label_slice[valid] = nerve_array[x_aug_original[valid, 0], x_aug_original[valid, 1], x_aug_original[valid, 2]]
            if label_slice.max() == 0: # 当前切片中没有神经，不用于训练
                continue
            nerve_slice_list.append(label_slice)

        image_slice[valid] = image_array[x_aug_original[valid, 0], x_aug_original[valid, 1], x_aug_original[valid, 2]]
        image_slice_list.append(image_slice)
        A_aug_dict["{:0>3d}".format(counter)] = A_aug.tolist()
        # A_aug_list.append({"{:0>3d}".format(counter): A_aug.tolist()})
        counter += 1
    
    return image_slice_list, nerve_slice_list, A_aug_dict

def main(image_path):
    # 1.分割下颌骨
    
    # 2.对下颌骨进行处理，提取牙弓曲线

    # 3.提取切片，临时储存切片与坐标变换矩阵

    # 4.关键点检测并还原到原空间

    # 5.拟合先验曲线

    # 6.分割网络进行分割
    pass


if __name__ == '__main__':
    image_folder = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/images' # 原始图像路径
    mandible_folder = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/labels/mandible'
    nerve_folder = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/labels/gt_alpha'
    img_slice_folder = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/imageSlices' # 切片储存路径
    nerve_slice_folder = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/nerveSlices'
    composition_folder = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/compos'
    json_path = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/slices/info.json' # 储存坐标变换矩阵的文件
    num_sample_pts = 200 # 牙弓曲线上切片的总数量

    # 训练模式
    train = True

    os.makedirs(img_slice_folder, exist_ok=True)
    os.makedirs(nerve_slice_folder, exist_ok=True)
    os.makedirs(composition_folder, exist_ok=True)

    filenames = os.listdir(image_folder)
    # filenames = ['1001012179_20180110.nii.gz']

    not_found = [] # 下颌骨或下颌神经缺失的文件
    filenames = tqdm(filenames)
    for filename in filenames:
        filenames.set_description("Processing %s" % filename)
        image_path = os.path.join(image_folder, filename)
        mandible_path = os.path.join(mandible_folder, filename)
        
        # 检查相关文件是否存在
        if train:
            nerve_path = os.path.join(nerve_folder, filename)
            if not os.path.exists(mandible_path) or not os.path.exists(nerve_path):  # 原始图像、下颌骨label、神经label必须同时存在
                print('Mandible or nerve file not found: ', filename)
                not_found.append(filename)
                continue
        else:
            nerve_path = None
            if not os.path.exists(mandible_path):
                print('Mandible file not found: ', filename)
                not_found.append(filename)
                continue
        
        # 提取切片
        img_slice_list, nerve_slice_list, A_aug_dict = extract_slices_new(image_path, mandible_path, nerve_path, num_sample_pts, train=train)
        # 导出切片
        for i in range(len(img_slice_list)):
            img = Image.fromarray(img_slice_list[i].astype(np.uint8))
            img.convert('L').save(os.path.join(img_slice_folder, filename[:-7]+'_'+"{:0>3d}".format(i)+'.png')) # 保存为单通道灰度图
            label = Image.fromarray(nerve_slice_list[i].astype(np.uint8) * 255) # 方便显示
            label.convert('1').save(os.path.join(nerve_slice_folder, filename[:-7]+'_'+"{:0>3d}".format(i)+'.png')) # 保存为二值图
            
            # 灰度图转化为RGB图
            rgb_image = np.stack((img_slice_list[i],)*3, axis=-1)
            mask = nerve_slice_list[i] == 1
            # 将mask区域标记为红色
            rgb_image[mask, 0] = 255
            rgb_image[mask, 1] = 0
            rgb_image[mask, 2] = 0
            composition = Image.fromarray(rgb_image.astype(np.uint8))
            composition.save(os.path.join(composition_folder, filename[:-7]+'_'+"{:0>3d}".format(i)+'.png'))

        # 导出坐标变换矩阵
        if not os.path.exists(json_path): # 创建json文件
            with open(json_path, 'w') as f:
                jsondata = {filename: A_aug_dict}
                json.dump(jsondata, f, indent=4, separators=(',', ': '))
        else:
            with open(json_path, 'r') as f: # 更新json文件，追加数据
                content = json.load(f)
                jsondata = {filename: A_aug_dict}
                content.update(jsondata)
            with open(json_path, 'w') as f:
                json.dump(content, f, indent=4, separators=(',', ': '))

    if len(not_found) > 0:
        print('Not found: ')
        print(*not_found, sep='\n')
    # img_slice_list, nerv_slice_list = extract_slices(image_path, mandible_path, nerve_path, nerveOnly=True)
    # plt.imshow(img_slice_list[0], cmap='gray')
    # plt.imsave('img_slice.jpeg', img_slice_list[0], cmap='gray')


