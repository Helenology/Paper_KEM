import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import SimpleITK as sitk
import tensorflow as tf
from tensorflow import keras
from keras.layers import AveragePooling3D

import time
import copy
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import os
import imageio
import pydicom as dicom
import pandas as pd
import sys

import SimpleITK
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
import cv2
import shutil

sys.path.append('../')
from utils import *


def load_experiment_data_lidc(patient_folder):
    """
    read data from LIDC dataset
    :param patient_folder: CT data's folder
    :return: CT data, slope, and intercept
    """
    # 因为LIDCdata里存放dcm的位置可能有.xml文件，没有的可以直接读取，
    # 但是有的话使用volread会报错，因此这里统一采用分开读的方式
    # Since LIDC folder has .xml file, directly using volread function will raise error
    # experiment_data = imageio.volread(PATIENT_FOLDER, format='DICOM')
    # thus we read each dcm file separately
    dcm_paths = glob(patient_folder + "*.dcm")  # find all .dcm files
    dcm_paths.sort()  # sort them
    dcm_file_list = []  # to place the CT slices
    for dcm_path in dcm_paths:
        dcm_file = dicom.dcmread(dcm_path)  # read the slice
        dcm_array = dcm_file.pixel_array  # get data
        dcm_file_list.append(dcm_array)  # record data
    # convert to np array
    experiment_data = np.stack(dcm_file_list, axis=0)
    experiment_data = experiment_data.astype(np.float32)
    # data rescaling parameters
    slope = float(dcm_file.RescaleSlope)
    intercept = float(dcm_file.RescaleIntercept)
    return experiment_data, slope, intercept


def transfer_range(image, CT_min, CT_max, slope=None, intercept=None):
    """
    :param: image: scale 0~1
    :param: CT_min: the minumum value of the original pixel_array in the dicom file
    :param: CT_max: the maximum value of the original pixel_array in the dicom file
    :param: slope: the slope for transfering the pixel_array to HU value
    :param: intercept: the intercept for tranfering the pixel_array to HU value
    return: the transfered image
    """
    pixel_array = image * (CT_max - CT_min) + CT_min
    if slope is not None:
        hu_image = pixel_array * slope + intercept  # transfer the range of the pixel_array to HU value range
        return hu_image

    return pixel_array


def get_patient_folder(path, index):
    """
    return the index th folder of the patient in LIDC dataset
    :param path: LIDC dataset's path
    :param index: the index th patient
    :return: patient's folder path
    """
    patient_folders = os.listdir(path)
    patient_folders = [path + item + '/' for item in patient_folders]
    patient_folder = patient_folders[index]
    patient_folder += os.listdir(patient_folder)[0] + '/'
    patient_folder += os.listdir(patient_folder)[0] + '/'
    return patient_folder


def get_segmented_lungs(aslice, lung_threshold):
    """
    maintain lung area's data in one slice and discard others
    :param aslice: a slice of the lung
    :param lung_threshold:
    :return: concentrated slice
    """
    binary = aslice <= lung_threshold    # 二值化图像
    cleared = clear_border(binary)   # 清除图像边界的小块区域
    label_image = label(cleared)  # 分割图像
    regions = regionprops(label_image)
    areas = [r.area for r in regions]  # 保留2个最大的联通区域
    areas.sort()
    if len(areas) > 2:
        for region in regions:
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    selem = disk(2)                         # Step 5: 图像腐蚀操作,将结节与血管剥离
    binary = binary_erosion(binary, selem)
    selem = disk(10)                        # Step 6: 图像闭环操作,保留贴近肺壁的结节
    binary = binary_closing(binary, selem)
#     edges = roberts(binary)                 # Step 7: 进一步将肺区残余小孔区域填充
#     binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0             # Step 8: 将二值化图像叠加到输入图像上
    aslice[get_high_vals] = 0
    return aslice


def lung_concentration(experiment_data, lung_threshold, slope, intercept):
    """
    maintain lung's data and discard others
    :param experiment_data: raw CT data
    :param lung_threshold: lung's threshold
    :param slope:
    :param intercept:
    :return: concentrated CT data
    """
    img = copy.copy(experiment_data)
    # rescale the lung_threshold to match the read CT data's range
    lung_threshold_m = (lung_threshold - intercept) / slope
    # lung concentration slice by slice
    for i in range(img.shape[0]):
        aslice = get_segmented_lungs(img[i], lung_threshold_m)
        img[i] = aslice
    return img


def transfer_pi_to_dicom(pi, original_folder, output_folder, truncate_threshold=None, CT_min=0, CT_max=4000):
    """
    transfer the value range of pi(0~1) to (CT_min~CT_max) and truncate if needed
    :param pi: with shape=(depth, 512, 512) and range(0~1) is the estimated probability of a voxel being vessel
    :param original_folder: the original patient folder, in which are all dicom files
    :param output_folder: the folder to dump all the processed dicom files
    :param truncate_threshold: range(0~1)
        if truncate_threshold is None: then no truncation is made;
        otherwise:
            where pi>=truncate_threshold values are preserved
            where pi< truncate_threshold values are set to be CT_mins
    :return pi: the processed pi which is stored in dicom type
    """
    if os.path.exists(output_folder) is False:  # output_folder does not exist
        print(f"====create a new folder at {output_folder}====")
        os.mkdir(output_folder)
    else:
        print(f"====recreate the folder at {output_folder}====")
        shutil.rmtree(output_folder)  # delete the whole folder
        os.mkdir(output_folder)  # create the output folder

    if truncate_threshold is None:
        print(f"====truncate_threshold=None, thus no truncation is made====")
    else:
        print(f"====truncate_threshold={truncate_threshold}, thus a truncation is made====")
        pi = (pi >= truncate_threshold) * (pi)

    # rescale the pi back to its original data range
    print(f"====change the pi range to ({CT_min}, {CT_max})====")
    pi = pi * (CT_max - CT_min) + CT_min
    pi = pi.astype(np.uint16)

    print(f"====read the original dicom files and prepare to replace the pixel_array====")
    dcm_files = glob(original_folder + "*.dcm")  # get a list of dicom paths
    dcm_files.sort()  # arange the path in the acsending order
    print(f"dcm_files length: {len(dcm_files)}")
    for i, dcm_file in enumerate(dcm_files):
        dcm_file = dcm_files[i]
        dcm = dicom.dcmread(dcm_file)  # read the dicom file
        dcm.pixel_array.flat = pi[i]  # the i th slice
        dcm.PixelData = dcm.pixel_array.tobytes()  # record the replacement
        dcm.save_as(output_folder + dcm_file.split('/')[-1])  # save the slice in dicom type
    print(f"====successfully save the dicom files at {output_folder}====")
    return pi


def experiment_compare_SPE(stats_dict):
    # information
    index = stats_dict['index']
    K = stats_dict["K"]
    lung_threshold = stats_dict["lung_threshold"]
    training_ratio = stats_dict["training_ratio"]
    Ch = stats_dict["Ch"]
    kernel_shape = stats_dict["kernel_shape"]
    to_csv_path = stats_dict["to_csv_path"]
    
    # patient CT
    results = [index]
    patient_folder = get_patient_folder(path, index)  # the patient's folder
    results.append(patient_folder)
    
    # read in experiment data
    experiment_data, slope, intercept = load_experiment_data_lidc(patient_folder)
    t1 = time.time()
    concentrated_data = lung_concentration(experiment_data, lung_threshold, slope, intercept)
    t2 = time.time()
    results.append(t2 - t1)
    
    # rescale the data
    CT_min = experiment_data.min()
    CT_max = experiment_data.max()
    concentrated_data = (concentrated_data - CT_min) / (CT_max - CT_min)
    shape = concentrated_data.shape
    experiment_data = tf.cast(tf.convert_to_tensor(concentrated_data), tf.float32)
    experiment_data = tf.reshape(experiment_data, (1,) + experiment_data.shape + (1,))
    
    # generate training data and testing data
    position_mask = np.random.binomial(n=1, p=training_ratio, size=experiment_data.shape)
    position_mask = tf.convert_to_tensor(position_mask, dtype=tf.float32)
    training_data = position_mask * experiment_data
    testing_data = (1.0 - position_mask) * experiment_data
    
    # KEM method
    print("\n-----------------KEM-----------------")
    sample_size = tf.reduce_sum(tf.cast(position_mask == 1, dtype=tf.float32)).numpy()
    bandwidth_base = sample_size**(-1/7)
    optimal_bandwidth = Ch * bandwidth_base
    kem_model = KEM_EXPE(K=K, shape=shape, 
                      training_data=training_data, 
                      position_mask=position_mask, 
                      kernel_shape=kernel_shape, 
                      bandwidth=optimal_bandwidth, 
                      kmeans_sample_ratio=1/100/training_ratio,
                      testing_data=testing_data)
    kem_model.kem_algorithm(max_steps=10, epsilon=1e-4, smooth_parameter=1e-20)
    if training_ratio < 1.0:
        kem_spe = kem_model.compute_prediction_error()
        results.append(kem_spe)
        
    # kmeans method
    print("\n-----------------kmeans-----------------")
    kmeans_model = Kmeans(K=K,
                      shape=shape, 
                      training_data=training_data, 
                      position_mask=position_mask, 
                      kmeans_sample_ratio=1/100/training_ratio,
                      testing_data=testing_data)
    kmeans_model.kmeans_algorithm(max_steps=10)
    if training_ratio < 1.0:
        kmeans_spe = kmeans_model.compute_prediction_error()
        results.append(kmeans_spe)
    
    # GMM method
    print("\n-----------------GMM-----------------")
    gmm_model = GMM(K=K, 
                shape=shape, 
                training_data=training_data, 
                position_mask=position_mask, 
                kmeans_sample_ratio=1/100/training_ratio,
                testing_data=testing_data)
    gmm_model.gmm_algorithm(max_steps=10, epsilon=1e-4, smooth_parameter=1e-20)
    if training_ratio < 1.0:
        gmm_spe = gmm_model.compute_prediction_error()
        results.append(gmm_spe)
    with open(to_csv_path, 'a', newline='', encoding='utf-8') as f:
        csv_write = csv.writer(f)  
        csv_write.writerow(results)
        

def transfer_pi_to_dicom_raw(vessel, original_folder, output_folder, truncate_threshold=None):
    if os.path.exists(output_folder) is False:  # output_folder does not exist
        print(f"====create a new folder at {output_folder}====")
        os.mkdir(output_folder)
    else:
        print(f"====recreate the folder at {output_folder}====")
        shutil.rmtree(output_folder)  # delete the whole folder
        os.mkdir(output_folder)  # create the output folder
    
    vessel_uint = vessel.astype(np.uint16)
    print(f"====read the original dicom files and prepare to replace the pixel_array====")
    dcm_files = glob(original_folder + "*.dcm")  # get a list of dicom paths
    dcm_files.sort()  # arange the path in the acsending order
    for i, dcm_file in enumerate(dcm_files):
        dcm_file = dcm_files[i]
        dcm = dicom.dcmread(dcm_file)  # read the dicom file
        dcm.pixel_array.flat = vessel_uint[i]  # the i th slice
        dcm.PixelData = dcm.pixel_array.tobytes()  # record the replacement
        dcm.save_as(output_folder + dcm_file.split('/')[-1])  # save the slice in dicom type
    print(f"====successfully save the dicom files at {output_folder}====")
    return vessel_uint

