# 实验 experiment
# sigma2_estimate = tf.nn.conv3d(self.pik_estimate * self.position_mask * (self.training_data - self.mu_estimate)**2, self.kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
# 使用的是self.mu_estimate，相当于是上一步的估计，不是mu_estimate当前步的估计

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

sys.path.append('../')
from utils import *

class KEM_EXPE():
    def __init__(self, K, shape,
                 training_data,
                 position_mask,
                 kernel_shape,
                 bandwidth=10 / 512,
                 kmeans_sample_ratio=1 / 100,
                 testing_data=None):
        # 将position_mask转化为tensor，并和training_data的大小对齐
        self.position_mask = position_mask
        self.K = K
        self.shape = shape
        self.training_data = training_data
        self.testing_data = testing_data
        print(f"From function(KEM_SIMU.__init__): training_data.shape: {self.training_data.shape}")
        self.pik_estimate = tf.ones((self.K, *shape, 1)) / self.K  # shape=(K, depth, height, width, 1)
        self.pi_estimate = tf.ones((self.K, *shape, 1)) / self.K  # equal prior probability initialization
        self.sigma_estimate = tf.ones((self.K, *shape, 1)) * 0.1  # standard deviation initialization
        self.mu_estimate = tf.ones((self.K, *shape, 1))

        # kmeans 挑选 mu_estimate 的初始值
        print(f"From function(__init__): Initialize mu via kmeans(with K={self.K})")
        t1 = time.time()
        x = self.training_data[self.position_mask > 0.5]
        x = tf.reshape(x, [-1, 1])
        random_x_sample_index = np.random.binomial(n=1, p=kmeans_sample_ratio, size=x.shape[0])
        random_x_sample = x[random_x_sample_index == 1]
        print(f"Randomly pick {random_x_sample_index.sum() / x.shape[0]} positions for kmeans.")
        model = KMeans(n_clusters=self.K)
        model.fit(random_x_sample)
        centers = model.cluster_centers_
        centers = centers.reshape((3,))
        centers = sorted(centers, reverse=True)
        print(centers)
        print(f"From function(kmeans_initialization): rearanged centers: {centers}")
        centers = tf.reshape(tf.cast(tf.constant(centers), tf.float32), [self.K, 1, 1, 1, 1])
        self.centers = centers
        t2 = time.time()
        print(
            f"From function(__init__): KMeans(with K={self.K}) success, with time: {t2 - t1:.4} seconds\n\tcenters: {tf.squeeze(self.centers)}")
        print("From function(__init__): Initialize parameters successfully.")
        print(
            f"\tpik_estimate:{self.pik_estimate.shape}\n\tpi_estimate: {self.pi_estimate.shape}\n\tmu_estimate: {self.mu_estimate.shape}\n\tsigma_estimate: {self.sigma_estimate.shape}")

        self.mu_estimate *= self.centers
        self.current_steps = 0
        # kernel，用于后续的convolution 加权求和
        self.kernel = self.generate_kernel(kernel_shape, bandwidth)
        print("From function(__init__): Initialize kernel successfully.")
        print(f"\tkernel: {self.kernel.shape}")

    def kem_algorithm(self, max_steps, epsilon, smooth_parameter=1e-5):
        print(f"From function(kem_algorithm): Receive max_steps: {max_steps}.")
        FLAG_CONTINUE = True
        self.denominator = tf.nn.conv3d(self.position_mask, self.kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        # self.denominator 需要出现在分母上，因此不能为0
        assert tf.reduce_sum(tf.cast(self.denominator == 0, dtype=tf.float32)) == 0

        while FLAG_CONTINUE:
            t1 = time.time()
            print(f"########################## STEP {self.current_steps} ##########################")
            # In each iteration, run e step and m step
            self.e_step(smooth_parameter)
            print(f"From function(kem_algorithm): E step success.")
            difference = self.m_step(smooth_parameter)
            print(f"From function(kem_algorithm): M step success.")

            # 迭代停止条件
            print(f"From function(kem_algorithm): difference: {difference:.6}.")
            if self.current_steps >= max_steps or np.isnan(difference) or difference < epsilon:
                FLAG_CONTINUE = False

            self.current_steps += 1
            t2 = time.time()
            print(f"---This iteration step costs {t2 - t1:.4} seconds.---")

    def e_step(self, smooth_parameter):
        # update parameters in the e step
        pik_estimate = normal_density_function_tf((self.training_data - self.mu_estimate) / self.sigma_estimate) * (
                    self.pi_estimate / self.sigma_estimate)
        pik_estimate_sum = tf.reduce_sum(pik_estimate, axis=0)
        # pik_estimate_sum 需要出现在分母，因此不能为0
        if tf.reduce_sum(tf.cast(pik_estimate_sum == 0, tf.float32)) > 0:
            print(f"+++ From m_step: add smooth_parameter to pik_estimate")
            pik_estimate += smooth_parameter
            pik_estimate_sum = tf.reduce_sum(pik_estimate, axis=0)
        pik_estimate /= pik_estimate_sum
        pik_difference = tf.reduce_mean((self.pik_estimate - pik_estimate) ** 2)
        print(f"\t Current pik difference: {pik_difference:.6}")
        self.pik_estimate = pik_estimate

    def m_step(self, smooth_parameter):
        # convolotion operations
        pi_estimate = tf.nn.conv3d(self.pik_estimate * self.position_mask, self.kernel, strides=[1, 1, 1, 1, 1],
                                   padding="SAME")
        # pi_estimate 需要出现在分母，因此不能为0
        if tf.reduce_sum(tf.cast(pi_estimate == 0, tf.float32)) > 0:
            print(f"+++ From m_step: add smooth_parameter to pi_estimate")
            pi_estimate += smooth_parameter
        mu_estimate = tf.nn.conv3d(self.pik_estimate * self.training_data * self.position_mask, self.kernel,
                                   strides=[1, 1, 1, 1, 1], padding="SAME")
        mu_estimate /= pi_estimate
        sigma2_estimate = tf.nn.conv3d(
            self.pik_estimate * self.position_mask * (self.training_data - self.mu_estimate) ** 2, self.kernel,
            strides=[1, 1, 1, 1, 1], padding="SAME")
        sigma_estimate = tf.sqrt(sigma2_estimate / pi_estimate)
        # sigma_estimate 需要出现在分母，因此不能为0
        if tf.reduce_sum(tf.cast(sigma_estimate == 0, tf.float32)) > 0:
            print(f"From m_step: add smooth_parameter to sigma_estimate")
            sigma_estimate += smooth_parameter

        pi_estimate /= self.denominator
        pi_estimate /= tf.reduce_sum(pi_estimate, axis=0)

        # 计算difference
        pi_difference = tf.reduce_mean((self.pi_estimate - pi_estimate) ** 2)
        print(f"\t Current pi difference: {pi_difference:.6}")
        mu_difference = tf.reduce_mean((self.mu_estimate - mu_estimate) ** 2)
        print(f"\t Current mu difference: {mu_difference:.6}")
        sigma_difference = tf.reduce_mean((self.sigma_estimate - sigma_estimate) ** 2)
        print(f"\t Current sigma difference: {sigma_difference:.6}")
        difference = pi_difference + mu_difference + sigma_difference
        # Update the estimators
        self.pi_estimate = pi_estimate
        self.mu_estimate = mu_estimate
        self.sigma_estimate = sigma_estimate

        return difference.numpy()

    def generate_kernel(self, kernel_shape, bandwidth):
        """
        Generate the kernel for the following convolotion operations.
        :param kernel_shape: the shape of the kernel (truncated)
        :param bandwidth: the bandwidth used for the kernel weight
        :return:
        """
        kernel = np.zeros(kernel_shape)
        center = int((kernel_shape[0]) / 2)
        center_coordinate = (center, center, center)
        # iterate each voxel space and compute the kernel weight
        for i in range(kernel_shape[0]):
            for j in range(kernel_shape[1]):
                for k in range(kernel_shape[2]):
                    kernel[i, j, k] = multivariate_kernel_function((i, j, k), center_coordinate, bandwidth, self.shape)
        kernel = tf.constant(kernel, dtype=tf.float32)
        # reshape kernel by [filter_depth, filter_height, filter_width, in_channels,out_channels]
        kernel = tf.reshape(kernel, shape=(*kernel_shape, 1, 1))
        # kernel 归一化，把权重的和调整为1
        kernel /= tf.reduce_sum(kernel)
        return kernel

    ### computing Yi_hat
    def compute_prediction_error(self):
        testing_position_mask = 1 - self.position_mask
        testing_size = tf.reduce_sum(testing_position_mask)
        Y = tf.squeeze(self.testing_data)  # [depth, height, width]
        predict_Y = tf.reduce_sum(tf.squeeze(self.pi_estimate * self.mu_estimate * testing_position_mask), axis=0)
        prediction_error = tf.reduce_sum((Y - predict_Y) ** 2) / testing_size
        prediction_error = prediction_error.numpy()
        return prediction_error