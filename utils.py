# Update on 2024.1.14
import matplotlib.pyplot as plt
import matplotlib
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from scipy.ndimage import zoom


def plot_3d(image, level, alpha=0.1, step_size=4):
    """
    Plot medical images in 3D structure. 用于绘制3D医学图像.
    :param image (np.array): 3 dimensional array in (z, y, x) order
    :param level (float): a value determine the value for the surface
    :param alpha (float): optional
                the level of transparency
    :param step_size (int): optional
                Step size in voxels. Default 1. Larger steps yield faster but
                coarser results. The result will always be topologically correct though.
    :return: nothing
    """
    # >level will be inside the surface; =level will be on the surface; <level will be outside the surface
    verts, faces, norm, val = measure.marching_cubes(image, level=level, step_size=step_size, allow_degenerate=True)
    # 3D plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # create a collection of 3D polygons, while alpha controls the transparency
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    # set color
    face_color = [0.45, 0.45, 0.75]  # color
    mesh.set_facecolor(face_color)
    # plot the 3D structure
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()


# 图片读入函数
# 读入数据，如有需要则翻转，输出图像矩阵、质心坐标、步长、是否翻转信息
def load_itk_image(filename):
    """
    Read in CT data from LUNA16 dataset.
    :param filename: the path of a CT file
    :return:
        numpyimage: the CT data in numpy array form
        origin: coordinates
        spacing: spacing of the CT data
        isflip: if the CT images are flipped, True; otherwise, False
    """
    with open(filename) as f:
        contents = f.readlines()  # basic information on the CT file; for example: DimSize = 512 512 201
        line = [k for k in contents if k.startswith('TransformMatrix')][0]  # find TransformMatrix = ??
        transform = np.array(line.split(' = ')[1].split(' ')).astype('float')  # the ?? part above
        transform = np.round(transform)  # convert the data type to int
        if np.any(transform != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):  # represents the CT data is flipped
            isflip = True
        else:
            isflip = False
    itkimage = sitk.ReadImage(filename)  # read the CT data
    numpyimage = sitk.GetArrayFromImage(itkimage)  # get CT data in numpy array form in (z, y, x) order

    if isflip is True:
        numpyimage = numpyimage[:, ::-1, ::-1]  # flip the CT data
    origin = np.array(itkimage.GetOrigin())  # in (x, y, z) order
    origin = np.array([origin[2], origin[1], origin[0]])  # in (z, y, x) order
    spacing = np.array(itkimage.GetSpacing())  # in (x, y, z) order
    spacing = np.array([spacing[2], spacing[1], spacing[0]])  # in (z, y, x) order

    return numpyimage, origin, spacing, isflip


def normal_density_function_np(x):
    """
    The value of the standard normal density function at x
    :param x: the value to evaluate for the density function
    :return: the value of the density function
    """
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)


def normal_density_function_tf(x):
    """
        The value of the standard normal density function at x
        :param x: the value to evaluate for the density function
        :return: the value of the density function
    """
    return 1 / tf.sqrt(2 * np.pi) * tf.exp(-x ** 2 / 2)


def multivariate_kernel_function(x1, x2, h, shape):
    """
    Compute the kernel weight.
    :param x1: location of an observation (i, j, k).
    :param x2: location of interest (i', j', k')
    :param h: bandwidth
    :param shape: the shape of the CT data with (depth, width, height)
    :return: kernel_weight
    """
    weight = 1
    for i in range(len(x1)):  # x1 is a 3D array
        weight *= normal_density_function_np((x1[i] - x2[i]) / shape[i] / h)
    return weight


# 由于不同图像的sampling不一样（voxel的大小不统一，这里统一为[1,1,1]的cubic
def resample(img, spacing, new_spacing=[1, 1, 1]):
    # 原来spacing=2.5mm，切片有4个，则一共占据10mm
    # 重采样，将spacing改为[1, 1, 1]，仍然占据10mm，但是spacing改变之后切片数量变为10个
    new_shape = img.shape * spacing / new_spacing  # 改变spacing后新图的大小，对每个维度 0，1，2 --> z, y, x
    resize_factor = new_shape / img.shape  # 新图尺寸 / 原图尺寸，即缩放比例，例如原像素间隔为2.5，新像素间隔为1，放缩比例为1/2.5
    new_img = zoom(img, zoom=resize_factor, mode='nearest')  # 放缩，边缘使用最近邻，插值默认为三线性插值
    return new_img


def truncate(img, truncate_shape):
    """
    Truncate the original image to truncate_shape.
    :param img: the original image in (z, x, y) order
    :param truncate_shape: the output shape of the image
    :return: truncate image
    """
    # 首先声明，截取的shape比原图小
    shape = img.shape
    assert shape > truncate_shape

    # 根据图像的中心点，上下左右截取truncate_shape大小的立方块
    min_coordinate = []
    max_coordinate = []
    for i in range(3):
        center_i = int(shape[i] / 2)
        front_i = int(truncate_shape[i] / 2)
        rear_i = truncate_shape[i] - front_i
        min_coordinate.append(center_i - front_i)
        max_coordinate.append(center_i + rear_i)
    truncate_img = img[min_coordinate[0]:max_coordinate[0],
                   min_coordinate[1]:max_coordinate[1],
                   min_coordinate[2]:max_coordinate[2]]
    assert truncate_img.shape == truncate_shape
    return truncate_img


def show_slice(aslice, title=None, Bar=True, OutLines=True, Bar_range=None, cmap=plt.cm.bone_r):
    """
    Show a slice through matplotlib.pyplot.
    :param aslice: the slice to be shown as an image
    :param title: default is None; if title is not None, then it is set as the title
    :param Bar: default is True, a colorbar will be plotted; otherwise, the colorbar will not be shown
    :param OutLines: default is True, a black frame will be added; othwise, the black frame will not be added
    :return: nothing
    """
    ############################## Normalize the colorkey
    # set the colorkey to a specific range at [Bar_range]
    if Bar_range is not None and len(Bar_range) == 2:
        vmin = min(Bar_range)
        vmax = max(Bar_range)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # no need to set the colorkey range; keep it default
    else:
        norm = None

    ############################## plot the image
    plt.imshow(aslice, cmap=cmap, norm=norm)  # norm的操作是把颜色固定在特定范围内，保证不同子图绘制出来的颜色是对齐的

    ############################## add a colorbar
    if Bar:
        plt.colorbar()

        ############################## outframe
    if OutLines:  # black frame is maintained
        plt.xticks([])
        plt.yticks([])
    else:  # the black frame is removed
        plt.axis('off')

    ############################## title
    if title is not None:
        plt.title(title)

    plt.show()


def bandwidth_preparation(CT_shape, position_mask, Ch):
    """
    prepare the bandwidth and kernel shape
    :param CT_shape: the shape of CT data
    :param position_mask: =1 is the training data; otherwise =0 is the testing data
    :param Ch: bandwidth constant
    :return: bandwidth and kernel shape
    """
    N = CT_shape[0] * CT_shape[1] * CT_shape[2]  # the total number of voxel positions
    sample_size = (position_mask == 1).numpy().sum()  # sample size used for the following KEM algorithm
    bandwidth_base = sample_size ** (-1 / 7)
    bandwidth = Ch * bandwidth_base
    kernel_size = int(bandwidth * 512)
    if kernel_size % 2 == 0:
        kernel_size -= 1  # adjust to odd number
    kernel_shape = (kernel_size,) * 3
    return bandwidth, kernel_shape


def bandwidth_preparation_small(position_mask, Ch):
    """
    prepare the bandwidth and kernel shape
    :param position_mask: =1 is the training data; otherwise =0 is the testing data
    :param Ch: bandwidth constant
    :return: bandwidth and kernel shape
    """
    sample_size = (position_mask == 1).numpy().sum()  # sample size used for the following KEM algorithm
    bandwidth_base = sample_size ** (-1 / 7)
    bandwidth = Ch * bandwidth_base
    kernel_size = int(bandwidth * 512)
    if kernel_size % 2 == 0:
        kernel_size -= 1  # adjust to odd number
    kernel_shape = (kernel_size,) * 3
    return bandwidth, kernel_shape


def bandwidth_preparation_big(position_mask, Ch, increase_kernel_size=1):
    """
    prepare the bandwidth and kernel shape
    :param position_mask: =1 is the training data; otherwise =0 is the testing data
    :param Ch: bandwidth constant
    :return: bandwidth and kernel shape
    """
    sample_size = (position_mask == 1).numpy().sum()  # sample size used for the following KEM algorithm
    bandwidth_base = sample_size ** (-1 / 7)
    bandwidth = Ch * bandwidth_base
    kernel_size = int(bandwidth * 512)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + increase_kernel_size  # adjust to odd number
    kernel_shape = (kernel_size,) * 3
    return bandwidth, kernel_shape


def compute_RMSE(kem, pi, mu, sigma):
    """
    compute RMSE of the estimate and parameters
    :param kem: kem class
    :param pi: real prior probability
    :param mu: real mean
    :param sigma: real standard deviation
    :return: rmse of pi, mu, and sigma
    """
    pi_estimate = tf.squeeze(kem.pi_estimate)
    pi_mse = tf.reduce_mean((pi_estimate - pi) ** 2)
    pi_mse = pi_mse.numpy()
    pi_rmse = np.sqrt(pi_mse)

    mu_estimate = tf.squeeze(kem.mu_estimate)
    mu_mse = tf.reduce_mean((mu_estimate - mu) ** 2)
    mu_mse = mu_mse.numpy()
    mu_rmse = np.sqrt(mu_mse)

    sigma_estimate = tf.squeeze(kem.sigma_estimate)
    sigma_mse = tf.reduce_mean((sigma_estimate - sigma) ** 2)
    sigma_mse = sigma_mse.numpy()
    sigma_rmse = np.sqrt(sigma_mse)

    return pi_rmse, mu_rmse, sigma_rmse


def compute_single_RMSE(estimator, parameter):
    """
    Compute RMSE of the estimator and parameter.
    :param estimator: estimator
    :param parameter: real parameter
    :return: rmse
    """
    mse = tf.reduce_mean((estimator - parameter) ** 2)
    mse = mse.numpy()
    rmse = np.sqrt(mse)
    return rmse

# def epanechnikov_kernel_function(x1, x2, h, shape):
#     """
#
#     :param x1:
#     :param x2:
#     :param h:
#     :param shape:
#     :return:
#     """
#     weight = 1
#     for i in range(len(x1)):
#         weight *= 0.75 * (1 - (x1[i] - x2[i]) / shape[i] / h)**2
#     return weight