import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import SimpleITK as sitk
import tensorflow as tf
from tensorflow import keras
from keras.layers import AveragePooling3D
import time
import copy
from sklearn.cluster import KMeans
import pandas as pd
import gc
from sklearn.linear_model import LinearRegression
import math
from multiprocessing import Process, Manager
import datetime
import sys
import csv
import os
from tensorflow.keras.utils import to_categorical

sys.path.append('../')
from utils import *


def get_original_data_newversion(lung_image_path, lung_mask_path, sample_CT_path):
    """
    Get CT data and its lung mask.
    :param lung_image_path: the path where all the CT files are placed
    :param lung_mask_path: the path where all the lung masks are placed
    :param sample_CT_path: the specific path of the CT data
    :return: the CT data (sample_CT_array) and the lung mask (lung_mask_array)
    """
    # read the CT data from sample_CT_path
    sample_CT_array, origin, spacing, isflip = load_itk_image(sample_CT_path)  # sample_CT_array in (z, y, x) order

    # read the corresponding lung mask
    lung_mask = sitk.ReadImage(sample_CT_path.replace(lung_image_path, lung_mask_path))
    lung_mask_array = sitk.GetArrayFromImage(lung_mask)  # in (z, y, x) order

    # The lung masks from LUNA16 annotates the left lung as 3 and the right lung as 4.
    # We set the lung part as 1 and others as 0
    lung_mask_array[lung_mask_array > 1] = 1
    if isflip is True:
        lung_mask_array = lung_mask_array[:, ::-1, ::-1]  # flip the lung mask if needed
    if sample_CT_array.shape != lung_mask_array.shape:
        print(sample_CT_array.shape, lung_mask_array.shape)
        assert sample_CT_array.shape == lung_mask_array.shape
    return sample_CT_array, lung_mask_array


def generate_pi_new_version(sample_CT_array, lung_mask_array, bone_threshold=400, IsShow=False):
    """
    Generate the prior probability of different classes and voxel space.
    :param sample_CT_array: The CT data
    :param lung_mask_array: The lung mask with lung=1 and non-lung=0
    :param bone_threshold: HU value > bone_threshold is regarded as bone
    :param IsShow: default as False; otherwise will show example results
    :return: prior probability (pi_tf) and the real class (voxel_class_tf) in tensor form
    """
    K = 3                                   # the number of classes
    CT_shape = sample_CT_array.shape        # with shape (depth, width, height)
    pi_shape = (K,) + CT_shape              # with shape (K, depth, width, height)
    voxel_class = np.zeros(shape=pi_shape)  # the real classes of the voxels

    ##############################################################
    #                           voxel class                      #
    ##############################################################
    # class 1 (bone): if value >= bone_threshold, then voxel_class(bone)[1] = 1; otherwise voxel_class(bone)[1] = 0
    voxel_class[1] = 1 * (sample_CT_array >= bone_threshold)
    # class 2 (lung): if lung mask = 1, then voxel_class(lung)[2] = 1; otherwise voxel_class(lung)[2] = 0
    voxel_class[2] = lung_mask_array
    voxel_class[2][voxel_class[1] == 1] = 0  # avoid overlapping with the class 1(bone)
    # class 0 (background): if not == 1 in class 1 and class 2: then = 1
    voxel_class[0] = 1 - voxel_class[1] - voxel_class[2]
    assert np.all((voxel_class == 0) | (voxel_class == 1))  # each element is either 0 or 1

    ##############################################################
    #                              pi                            #
    ##############################################################
    voxel_class_tf = tf.reshape(tf.convert_to_tensor(voxel_class, dtype=tf.float32), pi_shape + (1,))
    # locally average the classes as the prior probability
    pi_tf = AveragePooling3D(pool_size=(3, 3, 3), strides=1, padding="same")(voxel_class_tf)
    smoothness = tf.reshape([0.6, 0.6, 0.6], (3, 1, 1, 1, 1))  # smoothing term
    pi_tf += smoothness                                        # add the smoothing term
    pi_tf /= tf.reduce_sum(pi_tf, axis=0)                      # adjust the sum of pi to be 1
    voxel_class_tf = tf.squeeze(voxel_class_tf)                # with shape (K, depth, width, height)
    pi_tf = tf.squeeze(pi_tf)                                  # with shape (K, depth, width, height)

    if IsShow:                                                 # graphically display the voxel class and pi
        fig, ax = plt.subplots(2, 3)
        fig.set_figwidth(10)
        fig.tight_layout()
        for k in range(K):
            # voxel_class plotting
            ax0 = ax[0, k].imshow(voxel_class_tf[k, 99])
            ax[0, k].set_title(f"voxel_class[{k}]")
            fig.colorbar(ax0, ax=ax[0, k])
            # pi plotting
            ax1 = ax[1, k].imshow(pi_tf[k, 99])
            ax[1, k].set_title(f"pi[{k}]")
            fig.colorbar(ax1, ax=ax[1, k])
    return pi_tf, voxel_class_tf


def generate_mu_sigma_new_version(sample_CT_array, voxel_class, IsShow=False):
    """
    Generate the mean and std for the simulation study.
    :param sample_CT_array: the CT data
    :param voxel_class: the real classes
    :param IsShow: default as False; otherwise show example results
    :return: mu and sigma in tensor form
    """
    K = 3                             # the number of classes (0 for background; 1 for bone; 2 for lung)
    mu_variation = 0.25               # mu's coefficient for location variation
    CT_shape = sample_CT_array.shape  # CT data' shape
    pi_shape = (K,) + CT_shape        # parameter shape

    # rescale CT data to 0~1
    CT_max = sample_CT_array.max().astype(np.float64)                 # the max value of the CT data
    CT_min = sample_CT_array.min().astype(np.float64)                 # the min value of the CT data
    sample_CT_array = (sample_CT_array - CT_min) / (CT_max - CT_min)  # rescale the range to 0~1

    # basic position variation
    z_coordinates = np.arange(0, float(CT_shape[0])).reshape((CT_shape[0], 1, 1))
    z_coordinates = z_coordinates / CT_shape[0] * np.pi * 8  # 8 \pi x1
    y_coordinates = np.arange(0, float(CT_shape[1])).reshape((1, CT_shape[1], 1))
    y_coordinates = y_coordinates / CT_shape[1] * np.pi * 8  # 8 \pi x2
    x_coordinates = np.arange(0, float(CT_shape[2])).reshape((1, 1, CT_shape[2]))
    x_coordinates = x_coordinates / CT_shape[2] * np.pi * 8  # 8 \pi x3
    coordinates_weight = np.sin(x_coordinates) * np.sin(y_coordinates) * np.sin(z_coordinates)

    ##############################################################
    #                               mu                           #
    ##############################################################
    mu = np.zeros(pi_shape)     # allocate space for mu
    # global average of mu in three classes
    mu_const0 = 1                                 # class others
    data1 = sample_CT_array[voxel_class[1] == 1]  # class bone: data of this class
    mu_const1 = np.mean(data1)                    # class bone: global average of mu
    data2 = sample_CT_array[voxel_class[2] == 1]  # class lung: data of this class
    mu_const2 = np.mean(data2)                    # class lung: global average of mu
    print(f"From function(generate_mu_sigma_new_version)")
    print(f"\t mu_const: Class bone[{mu_const1:.4f}]; Class lung[{mu_const2:.4f}]")
    # If the two mu constants are too close, they are adjusted below
    if abs(mu_const1 - mu_const2) < 0.4:  # if two means are too near, enlarge the gap
        mu_const2 = mu_const1 - 0.4
    # spatial variation for mean function
    mu[0] = mu_const0 + coordinates_weight * mu_variation  # add position variation
    mu[1] = mu_const1 + coordinates_weight * mu_variation  # add position variation
    mu[2] = mu_const2 + coordinates_weight * mu_variation  # add position variation

    ##############################################################
    #                           sigma                            #
    ##############################################################
    sigma = np.zeros(pi_shape)    # allocate space for sigma
    sigma_const1 = np.std(data1)  # class 1 (class bone): std
    sigma_const2 = np.std(data2)  # class 2 (class lung): std
    sigma[1] = abs(sigma_const1 + coordinates_weight * sigma_const1)
    sigma[2] = abs(sigma_const2 + coordinates_weight * sigma_const2)
    sigma[0] = sigma[1]           # set sigma for class others

    if IsShow:                    # graphically display the mu and sigma
        fig, ax = plt.subplots(2, 3)
        fig.set_figwidth(10)
        fig.tight_layout()
        for k in range(K):
            # mu plotting
            ax0 = ax[0, k].imshow(mu[k, 99])
            ax[0, k].set_title(f"mu[{k}]: {tf.reduce_mean(mu[k]):.4}")
            fig.colorbar(ax0, ax=ax[0, k])
            # sigma plotting
            ax1 = ax[1, k].imshow(sigma[k, 99])
            ax[1, k].set_title(f"sigma[{k}]: {tf.reduce_mean(sigma[k]):.4}")
            fig.colorbar(ax1, ax=ax[1, k])

    # convert array to tensor
    mu = tf.convert_to_tensor(mu, dtype=tf.float32)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    return mu, sigma


def generate_simulate_data_new_version(pi, mu, sigma, seed=0, IsShow=False):
    """
    Generate simulate data with prior probability pi, mean mu, and standard deviation sigma
    :param pi: the prior probability of K classes
    :param mu: the mean function
    :param sigma: the standard deviation function
    :param seed: random seed
    :param IsShow: default as False; otherwise will show the example results
    :return: simulate data in tensor form
    """
    # set random seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    pi_shape = pi.shape                                           # (K, depth, height, width)
    K = pi_shape[0]                                               # the number of classes
    CT_shape = pi_shape[1:]                                       # (depth, height, width)
    simulate_data = tf.zeros(shape=CT_shape)                      # allocate space for simulate data

    # get a realization based on probability of pi
    tmp_pi = tf.transpose(pi)                                     # (****, K=3)
    transpose_pi_shape = tmp_pi.shape
    tmp_pi = tf.reshape(tmp_pi, [-1, 3])                          # reshape the pi into 2-dimensional array
    pi_realization = tf.random.stateless_categorical(tf.math.log(tmp_pi), 1, seed=[0, 0])
    pi_realization = to_categorical(pi_realization)
    pi_realization = tf.reshape(pi_realization, transpose_pi_shape)
    pi_realization = tf.transpose(pi_realization)                 # transpose to be the original order
    assert np.all((pi_realization == 0) | (pi_realization == 1))  # each one-hot label is either 0 or 1

    pi_realization = tf.convert_to_tensor(pi_realization, dtype=tf.float32)       # a realization of the pi
    for k in range(K):
        tmp_data = tf.random.normal(shape=CT_shape, mean=mu[k], stddev=sigma[k])  # normal random variable
        if IsShow:  # graphically display the realization
            print(f"################ k = {k} ################")
            print(f"[{k}] tmp_data.max: {tf.reduce_max(tmp_data):.4}")
            print(f"[{k}] tmp_data.mean: {tf.reduce_mean(tmp_data):.4}")
            show_slice(pi_realization[k, 99], title=f"pi_realization for k={k}")
        simulate_data += tmp_data * pi_realization[k]  # if pi_realization=1, then is set to be normal r.v., otherwise 0

    if IsShow:  # graphically display the simulate data
        print(f"from function(generate_simulate_data_new_version): simulate_data.max={tf.reduce_max(simulate_data):.4}")
        show_slice(simulate_data[99], title=f"simulate_data[0, 99]")
    simulate_data = tf.reshape(simulate_data, shape=(1,) + CT_shape + (1,))
    return simulate_data, pi_realization
