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
import tensorflow as tf
from multiprocessing import Process, Manager
import datetime
import sys
import csv
import os
from tensorflow.keras.utils import to_categorical
from data_generate import *  # 参数生成文件
from KEM_SIMU import *  # KEM 类

sys.path.append('../')
sys.path.append('../Models/')
from utils import *
from GMM import *
from Kmeans import *


def compute_Ch_4_cv_reg(stats_dict, Ch_path, training_ratio=0.8):
    """
    compute the optimal bandwidth based on CV method and REG method
    :param stats_dict: basic information dict
    :param Ch_path: path to write the results
    :param training_ratio: used for training
    :return: nothing
    """
    t1 = time.time()
    # kernel sizes and bandwidth candidates
    # CV
    CV_kernel_size_list = stats_dict['CV_kernel_size_list']
    CV_bandwidth_list = stats_dict['CV_bandwidth_list']
    CV_coef_list = stats_dict['CV_coef_list']
    # REG
    REG_kernel_size_list = stats_dict['REG_kernel_size_list']
    REG_bandwidth_list = stats_dict['REG_bandwidth_list']

    # load the CT data
    index = stats_dict['index']
    sample_CT_path = stats_dict['lung_image_file_list'][index]
    sample_CT_array, lung_mask_array = get_original_data_newversion(stats_dict['lung_image_path'],
                                                                    stats_dict['lung_mask_path'], sample_CT_path)
    print(f"---------------WE ARE LOADING {index}th PATIENT's CT with shape {sample_CT_array.shape}---------------")

    # generating the true parameters [pi, mu, sigma]
    pi, voxel_class = generate_pi_new_version(sample_CT_array, lung_mask_array, bone_threshold=400)
    mu, sigma = generate_mu_sigma_new_version(sample_CT_array, voxel_class)
    # based on the true parameters generating simulation data
    simulate_data, pi_realization = generate_simulate_data_new_version(pi, mu, sigma, seed=0)
    shape = sample_CT_array.shape

    # generate position_mask for every bandwidth candidates
    position_mask = np.random.binomial(n=1, p=training_ratio, size=simulate_data.shape)
    position_mask = tf.convert_to_tensor(position_mask, dtype=tf.float32)

    # generating training data and testing data
    training_data = position_mask * simulate_data
    testing_data = (1 - position_mask) * simulate_data
    sample_size = (position_mask == 1).numpy().sum()
    bandwidth_base = sample_size ** (-1 / 7)

    # REG for bandwidth selection
    t2 = time.time()
    SPE_results_REG = []
    for i, kernel_size in enumerate(REG_kernel_size_list):
        bandwidth = REG_bandwidth_list[i]
        kernel_shape = (kernel_size,) * 3  # (kernel_size, kernel_size, kernel_size)
        Ch = bandwidth / bandwidth_base
        print(f"##########[{index}]########## The {i}th bandwidth from REG ####################")
        kem_reg = KEM_SIMU_complex(K=3, shape=shape,
                                   training_data=training_data,
                                   position_mask=position_mask,
                                   kernel_shape=kernel_shape,
                                   bandwidth=bandwidth,
                                   kmeans_sample_ratio=1 / (100 * training_ratio),
                                   testing_data=testing_data)
        # KEM Algorithm
        kem_reg.kem_algorithm(max_steps=20, epsilon=1e-4, smooth_parameter=1e-20)
        # SPE on the testing data
        SPE = kem_reg.compute_prediction_error()
        SPE_results_REG.append([i, kernel_size, bandwidth, SPE, Ch, Ch ** 4 / 4, Ch ** (-3)])

    # regression method for bandwidth constant selection
    SPE_results_DF = pd.DataFrame(SPE_results_REG,
                                  columns=["i", "kernel_size", "bandwidth", "SPE", "Ch", 'Ch^4/4', 'Ch^-3'])
    SPE_results_DF = SPE_results_DF.dropna(axis=0, how='any')
    model = LinearRegression()
    model = model.fit(SPE_results_DF.loc[:, ['Ch^4/4', 'Ch^-3']], SPE_results_DF.SPE)
    Ch_REG = math.pow((3 * (model.coef_[1]) / (model.coef_[0])), 1 / 7)

    # CV for bandwidth selection
    t3 = time.time()
    SPE_results_CV = []
    for i, kernel_size in enumerate(CV_kernel_size_list):
        bandwidth = CV_bandwidth_list[i]
        coef = CV_coef_list[i]
        kernel_shape = (kernel_size,) * 3  # (kernel_size, kernel_size, kernel_size)
        Ch = bandwidth / bandwidth_base
        print(f"##########[{index}]########## The {i}th bandwidth from CV ####################")
        kem_cv = KEM_SIMU_complex(K=3, shape=shape,
                                  training_data=training_data,
                                  position_mask=position_mask,
                                  kernel_shape=kernel_shape,
                                  bandwidth=bandwidth,
                                  kmeans_sample_ratio=1 / (100 * training_ratio),
                                  testing_data=testing_data)
        # KEM Algorithm
        kem_cv.kem_algorithm(max_steps=20, epsilon=1e-4, smooth_parameter=1e-20)
        # SPE on the testing data
        SPE = kem_cv.compute_prediction_error()
        SPE_results_CV.append([i, kernel_size, coef, SPE, Ch])

    # CV method for bandwidth constant selection - argmin SPE
    SPE_results_DF = pd.DataFrame(SPE_results_CV, columns=["i", "kernel_size", "coef", "SPE", "Ch"])
    SPE_results_DF = SPE_results_DF.dropna(axis=0, how='any')
    CV_min_results = SPE_results_DF.loc[SPE_results_DF.SPE.argmin(),]
    Ch_CV = CV_min_results["Ch"]
    t4 = time.time()
    i_CV = CV_min_results["i"]
    kernel_size_CV = CV_min_results["kernel_size"]
    coef_CV = CV_min_results["coef"]

    # record results
    with open(Ch_path, 'a', newline='', encoding='utf-8') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(
            [f"patient_{index}", t2 - t1, Ch_CV, i_CV, kernel_size_CV, coef_CV, t4 - t3, Ch_REG, t3 - t2,
             model.coef_[0], model.coef_[1]])


def compute_metrics_4_KEM(stats_dict, to_csv_path):
    """
    Compute metrics for the KEM method.
    :param stats_dict: basic information dictionary.
    :param to_csv_path: path to record the metrics.
    :return: nothing
    """
    index = stats_dict['index']
    Ch_CV = stats_dict['Ch_CV']
    Ch_REG = stats_dict['Ch_REG']
    sample_CT_path = stats_dict['lung_image_file_list'][index]
    sample_CT_array, lung_mask_array = get_original_data_newversion(stats_dict['lung_image_path'],
                                                                    stats_dict['lung_mask_path'], sample_CT_path)
    print(f"---------------WE ARE LOADING {index}th PATIENT's CT with shape {sample_CT_array.shape}---------------")
    # generating the true parameters [pi, mu, sigma]
    pi, voxel_class = generate_pi_new_version(sample_CT_array, lung_mask_array, bone_threshold=400)
    mu, sigma = generate_mu_sigma_new_version(sample_CT_array, voxel_class)

    # based on the true parameters generating simulation data
    simulate_data, pi_realization = generate_simulate_data_new_version(pi, mu, sigma, seed=0)
    shape = sample_CT_array.shape

    # according to the metrics, we decide on the ratio lists
    ratio_list = [0.8, 1.0]  # 0.8 is for test acc; 1.0 is for rmse
    results = []
    for training_ratio in ratio_list:
        print(f"#################### [Patient_{index}] ratio={training_ratio} ####################")
        # generate position_mask for every bandwidth candidates
        position_mask = np.random.binomial(n=1, p=training_ratio, size=simulate_data.shape)
        position_mask = tf.convert_to_tensor(position_mask, dtype=tf.float32)

        # generating training data
        training_data = position_mask * simulate_data
        testing_data = (1 - position_mask) * simulate_data

        # reduce voxel with 0,1,2 representing each class
        single_class = pi_realization[1] * 1 + pi_realization[2] * 2
        true_test_class = single_class[tf.squeeze(position_mask) < 0.5]

        sample_size = (position_mask == 1).numpy().sum()
        bandwidth_base = sample_size ** (-1 / 7)

        # CV Method
        opt_bwd_CV = Ch_CV * bandwidth_base
        kernel_size_CV = int(opt_bwd_CV * 512)
        if kernel_size_CV % 2 == 0:
            kernel_size_CV -= 1
        kernel_shape_CV = (kernel_size_CV,) * 3
        print(f"#################### [{index}] CV ####################")
        kem_cv = KEM_SIMU_complex(K=3, shape=shape,
                                  training_data=training_data,
                                  position_mask=position_mask,
                                  kernel_shape=kernel_shape_CV,
                                  bandwidth=opt_bwd_CV,
                                  kmeans_sample_ratio=1 / (100 * training_ratio),
                                  testing_data=testing_data)
        kem_cv.kem_algorithm(max_steps=20, epsilon=1e-4, smooth_parameter=1e-20)  # KEM Algorithm

        # REG Method
        opt_bwd_REG = Ch_REG * bandwidth_base
        kernel_size_REG = int(opt_bwd_REG * 512)
        if kernel_size_REG % 2 == 0:
            kernel_size_REG -= 1
        kernel_shape_REG = (kernel_size_REG,) * 3
        print(f"#################### [{index}] REG ####################")
        kem_reg = KEM_SIMU_complex(K=3, shape=shape,
                                   training_data=training_data,
                                   position_mask=position_mask,
                                   kernel_shape=kernel_shape_REG,
                                   bandwidth=opt_bwd_REG,
                                   kmeans_sample_ratio=1 / (100 * training_ratio),
                                   testing_data=testing_data)
        kem_reg.kem_algorithm(max_steps=20, epsilon=1e-4, smooth_parameter=1e-20)
        if training_ratio < 1:
            print(f"#################### [Patient_{index}] ACC ####################")
            CV_test_class = kem_cv.predict_test_class()
            CV_acc = tf.reduce_mean(tf.cast(true_test_class == CV_test_class, tf.float32))
            CV_acc = CV_acc.numpy()
            print(f"\tCV_ACC: {CV_acc:.4}")

            REG_test_class = kem_reg.predict_test_class()
            REG_acc = tf.reduce_mean(tf.cast(true_test_class == REG_test_class, tf.float32))
            REG_acc = REG_acc.numpy()
            results += [f"patiant_{index}", CV_acc, REG_acc]
            print(f"\tGMM_ACC: {REG_acc:.4}")

        elif training_ratio == 1:
            print(f"#################### [Patient_{index}] RMSE ####################")
            CV_pi_rmse, CV_mu_rmse, CV_sigma_rmse = compute_RMSE(kem_cv, pi, mu, sigma)  # CV RMSE
            results += [f"patient_{index}", CV_pi_rmse, CV_mu_rmse, CV_sigma_rmse]
            print(f"\tRMSE_CV: pi_rmse: {CV_pi_rmse:.4}; mu_rmse: {CV_mu_rmse:.4}; sigma_rmse: {CV_sigma_rmse:.4}")

            REG_pi_rmse, REG_mu_rmse, REG_sigma_rmse = compute_RMSE(kem_reg, pi, mu, sigma)
            results += [REG_pi_rmse, REG_mu_rmse, REG_sigma_rmse]
            print(f"\tRMSE_REG: pi_rmse: {REG_pi_rmse:.4}; mu_rmse: {REG_mu_rmse:.4}; sigma_rmse: {REG_sigma_rmse:.4}")

            with open(to_csv_path, 'a', newline='', encoding='utf-8') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(results)


# def compute_rmse_4_cv_reg(stats_dict, bdw_mse_path):
#     """
#     compute rmse based on optimal bandwidths of CV and REG method
#     :param stats_dict: basic information
#     :param bdw_mse_path: path to write the rmse results
#     :return: nothing
#     """
#     index = stats_dict['index']
#     sample_CT_path = stats_dict['lung_image_file_list'][index]
#     sample_CT_array, lung_mask_array = get_original_data_newversion(stats_dict['lung_image_path'],
#                                                                     stats_dict['lung_mask_path'], sample_CT_path)
#     print(f"---------------WE ARE LOADING {index}th PATIENT's CT with shape {sample_CT_array.shape}---------------")
#
#     # generating the true parameters [pi, mu, sigma]
#     pi, voxel_class = generate_pi_new_version(sample_CT_array, lung_mask_array, bone_threshold=400)
#     mu, sigma = generate_mu_sigma_new_version(sample_CT_array, voxel_class)
#
#     # based on the true parameters generating simulation data
#     simulate_data, pi_realization = generate_simulate_data_new_version(pi, mu, sigma, seed=0)
#     shape = sample_CT_array.shape
#     N = shape[0] * shape[1] * shape[2]
#     sample_size = N
#     training_ratio = 1
#
#     # generate position_mask for every bandwidth candidates
#     position_mask = np.random.binomial(n=1, p=training_ratio, size=simulate_data.shape)
#     position_mask = tf.convert_to_tensor(position_mask, dtype=tf.float32)
#
#     # generating training data
#     training_data = position_mask * simulate_data
#     # the data needed to rerun
#     tmp_df = stats_dict["tmp_df"]
#     tmp_df = tmp_df.iloc[stats_dict["i"], ]
#     mse_results = []
#
#     # CV
#     optimal_bandwidth_CV = tmp_df['bdw_CV']
#     kernel_size_CV = int(optimal_bandwidth_CV * 512)
#     if kernel_size_CV % 2 == 0:
#         kernel_size_CV -= 1
#     kernel_shape_CV = (kernel_size_CV,) * 3
#     print(f"#################### [{index}] CV ####################")
#     kem_cv = KEM_SIMU_complex(K=3, shape=shape,
#                               training_data=training_data,
#                               position_mask=position_mask,
#                               kernel_shape=kernel_shape_CV,
#                               bandwidth=optimal_bandwidth_CV,
#                               kmeans_sample_ratio=1 / (100 * training_ratio),
#                               testing_data=None)
#     # KEM Algorithm
#     kem_cv.kem_algorithm(max_steps=50, epsilon=1e-4, smooth_parameter=1e-20)
#     # CV MSE
#     pi_rmse, mu_rmse, sigma_rmse = compute_RMSE(kem_cv, pi, mu, sigma)
#     mse_results += [f"patient_{index}", pi_rmse, mu_rmse, sigma_rmse]
#     print(f"\tRMSE_CV: pi_rmse: {pi_rmse:.4}; mu_rmse: {mu_rmse:.4}; sigma_rmse: {sigma_rmse:.4}")
#
#     # REG
#     optimal_bandwidth_REG = tmp_df['bdw_REG']
#     kernel_size_REG = int(optimal_bandwidth_REG * 512)
#     if kernel_size_REG % 2 == 0:
#         kernel_size_REG += 1
#     kernel_shape_REG = (kernel_size_REG,) * 3
#     print(f"########## [{index}] REG##########")
#     kem_reg = KEM_SIMU_complex(K=3, shape=shape,
#                                training_data=training_data,
#                                position_mask=position_mask,
#                                kernel_shape=kernel_shape_REG,
#                                bandwidth=optimal_bandwidth_REG,
#                                kmeans_sample_ratio=1 / (100 * training_ratio),
#                                testing_data=None)
#     # KEM Algorithm
#     kem_reg.kem_algorithm(max_steps=50, epsilon=1e-4, smooth_parameter=1e-20)
#     # REG MSE
#     pi_rmse, mu_rmse, sigma_rmse = compute_RMSE(kem_reg, pi, mu, sigma)
#     mse_results += [pi_rmse, mu_rmse, sigma_rmse]
#     print(f"\tRMSE_REG: pi_rmse: {pi_rmse:.4}; mu_rmse: {mu_rmse:.4}; sigma_rmse: {sigma_rmse:.4}")
#
#     # record results
#     with open(bdw_mse_path, 'a', newline='', encoding='utf-8') as f:
#         csv_write = csv.writer(f)
#         csv_write.writerow(mse_results)


def compute_rmse_4_consistency(stats_dict, path, training_ratio_list, Ch):
    """
    compute RMSE for consistency
    :param stats_dict: basic information
    :param path: path to write rmse results
    :param training_ratio_list: sampling ratio list
    :param Ch: bandwidth constant
    :return: nothing
    """
    index = stats_dict['index']
    # read in CT data
    sample_CT_path = stats_dict['lung_image_file_list'][index]
    sample_CT_array, lung_mask_array = get_original_data_newversion(stats_dict['lung_image_path'],
                                                                    stats_dict['lung_mask_path'], sample_CT_path)
    # generate parameters
    pi, voxel_class = generate_pi_new_version(sample_CT_array, lung_mask_array, bone_threshold=400)
    mu, sigma = generate_mu_sigma_new_version(sample_CT_array, voxel_class)
    # generate simulate data
    simulate_data, pi_realization = generate_simulate_data_new_version(pi, mu, sigma, seed=0)
    CT_shape = sample_CT_array.shape

    for training_ratio in training_ratio_list:
        print(f"------------- Ratio {training_ratio} -------------")
        position_mask = np.random.binomial(n=1, p=training_ratio, size=simulate_data.shape)
        position_mask = tf.convert_to_tensor(position_mask, dtype=tf.float32)
        training_data = position_mask * simulate_data
        bandwidth, kernel_shape = bandwidth_preparation_small(position_mask, Ch)

        kem = KEM_SIMU_complex(K=3, shape=CT_shape,
                               training_data=training_data,
                               position_mask=position_mask,
                               kernel_shape=kernel_shape,
                               bandwidth=bandwidth,
                               kmeans_sample_ratio=1 / 100 / training_ratio,
                               testing_data=None)
        # KEM algorithm
        kem.kem_algorithm(max_steps=20, epsilon=1e-4, smooth_parameter=1e-20)
        # compute RMSE
        pi_rmse, mu_rmse, sigma_rmse = compute_RMSE(kem, pi, mu, sigma)
        print(f"\tUltimate MSE: pi_rmse: {pi_rmse:.4}; mu_rmse: {mu_rmse:.4}; sigma_rmse: {sigma_rmse:.4}")
        # record results
        mse_results = [f"patient_{index}", CT_shape[0], training_ratio, kernel_shape[0], pi_rmse, mu_rmse, sigma_rmse]
        with open(path, 'a', newline='', encoding='utf-8') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(mse_results)


def compute_metrics_4_GMM_Kmeans(stats_dict, to_csv_path):
    """
    Compute metrics for the GMM and Kmeans methods.
    :param stats_dict: basic information dictionary.
    :param to_csv_path: path to record the metrics.
    :return: nothing
    """
    index = stats_dict['index']
    sample_CT_path = stats_dict['lung_image_file_list'][index]
    sample_CT_array, lung_mask_array = get_original_data_newversion(stats_dict['lung_image_path'],
                                                                    stats_dict['lung_mask_path'], sample_CT_path)
    print(f"---------------WE ARE LOADING {index}th PATIENT's CT with shape {sample_CT_array.shape}---------------")
    # generating the true parameters [pi, mu, sigma]
    pi, voxel_class = generate_pi_new_version(sample_CT_array, lung_mask_array, bone_threshold=400)
    mu, sigma = generate_mu_sigma_new_version(sample_CT_array, voxel_class)

    # based on the true parameters generating simulation data
    simulate_data, pi_realization = generate_simulate_data_new_version(pi, mu, sigma, seed=0)

    # according to the metrics, we decide on the ratio lists
    ratio_list = [0.8, 1.0]  # 0.8 is for test acc; 1.0 is for rmse
    results = []
    for training_ratio in ratio_list:
        print(f"#################### [Patient_{index}] ratio={training_ratio} ####################")
        # generate position_mask for every bandwidth candidates
        position_mask = np.random.binomial(n=1, p=training_ratio, size=simulate_data.shape)
        position_mask = tf.convert_to_tensor(position_mask, dtype=tf.float32)

        # generating training data
        training_data = position_mask * simulate_data
        testing_data = (1 - position_mask) * simulate_data

        # reduce voxel with 0,1,2 representing each class
        single_class = pi_realization[1] * 1 + pi_realization[2] * 2
        true_test_class = single_class[tf.squeeze(position_mask) < 0.5]

        # Kmeans
        kmeans_model = Kmeans(K=3,
                              shape=sample_CT_array.shape,
                              training_data=training_data,
                              position_mask=position_mask,
                              kmeans_sample_ratio=1 / 100 / training_ratio,
                              testing_data=testing_data)
        kmeans_model.kmeans_algorithm(max_steps=20)
        # GMM
        gmm_model = GMM(K=3,
                        shape=sample_CT_array.shape,
                        training_data=training_data,
                        position_mask=position_mask,
                        kmeans_sample_ratio=1 / 100 / training_ratio,
                        testing_data=testing_data)
        gmm_model.gmm_algorithm(max_steps=20, epsilon=1e-4, smooth_parameter=1e-20)

        if training_ratio < 1.0:
            print(f"#################### [Patient_{index}] ACC ####################")
            kmeans_test_class = kmeans_model.predict_test_class()
            kmeans_acc = tf.reduce_mean(tf.cast(true_test_class == kmeans_test_class, tf.float32))
            kmeans_acc = kmeans_acc.numpy()
            print(f"\tKmeans_ACC: {kmeans_acc:.4}")

            gmm_test_class = gmm_model.predict_test_class()
            gmm_acc = tf.reduce_mean(tf.cast(true_test_class == gmm_test_class, tf.float32))
            gmm_acc = gmm_acc.numpy()
            results += [f"patiant_{index}", kmeans_acc, gmm_acc]
            print(f"\tGMM_ACC: {gmm_acc:.4}")

        elif training_ratio == 1.0:
            print(f"#################### [Patient_{index}] RMSE ####################")
            kmeans_pi_estimate = kmeans_model.pi_estimate.reshape(3, 1, 1, 1)
            kmeans_mu_estimate = kmeans_model.mu_estimate.reshape(3, 1, 1, 1)
            kmeans_sigma_estimate = kmeans_model.sigma_estimate.reshape(3, 1, 1, 1)
            kmeans_pi_rmse = compute_single_RMSE(kmeans_pi_estimate, pi)
            kmeans_mu_rmse = compute_single_RMSE(kmeans_mu_estimate, mu)
            kmeans_sigma_rmse = compute_single_RMSE(kmeans_sigma_estimate, sigma)
            print(f"\tKmeans_RMSE: pi_rmse: {kmeans_pi_rmse:.4}; mu_rmse: {kmeans_mu_rmse:.4};"
                  f" sigma_rmse: {kmeans_sigma_rmse:.4}\n")

            GMM_pi_estimate = tf.squeeze(gmm_model.pi_estimate, -1)
            GMM_mu_estimate = tf.squeeze(gmm_model.mu_estimate, -1)
            GMM_sigma_estimate = tf.squeeze(gmm_model.sigma_estimate, -1)
            GMM_pi_rmse = compute_single_RMSE(GMM_pi_estimate, pi)
            GMM_mu_rmse = compute_single_RMSE(GMM_mu_estimate, mu)
            GMM_sigma_rmse = compute_single_RMSE(GMM_sigma_estimate, sigma)
            results += [f"patient_{index}", kmeans_pi_rmse, kmeans_mu_rmse, kmeans_sigma_rmse, GMM_pi_rmse, GMM_mu_rmse,
                        GMM_sigma_rmse]
            print(
                f"\tGMM_RMSE: pi_rmse: {GMM_pi_rmse:.4}; mu_rmse: {GMM_mu_rmse:.4}; sigma_rmse: {GMM_sigma_rmse:.4}\n")

    # record results
    with open(to_csv_path, 'a', newline='', encoding='utf-8') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(results)


def compute_rmse_4_consistency_new(stats_dict, path, training_ratio_list, Ch, increase_kernel_size=-1):
    """
    compute RMSE for consistency
    :param stats_dict: basic information
    :param path: path to write rmse results
    :param training_ratio_list: sampling ratio list
    :param Ch: bandwidth constant
    :return: nothing
    """
    seed = stats_dict['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    index = stats_dict['index']

    # read in CT data
    sample_CT_path = stats_dict['lung_image_file_list'][index]
    sample_CT_array, lung_mask_array = get_original_data_newversion(stats_dict['lung_image_path'],
                                                                    stats_dict['lung_mask_path'], sample_CT_path)
    # generate parameters
    pi, voxel_class = generate_pi_new_version(sample_CT_array, lung_mask_array, bone_threshold=400)
    mu, sigma = generate_mu_sigma_new_version(sample_CT_array, voxel_class)

    # generate simulate data
    simulate_data, pi_realization = generate_simulate_data_new_version(pi, mu, sigma, seed=seed)
    CT_shape = sample_CT_array.shape

    for training_ratio in training_ratio_list:
        print(f"------------- Ratio {training_ratio} -------------")
        position_mask = np.random.binomial(n=1, p=training_ratio, size=simulate_data.shape)
        position_mask = tf.convert_to_tensor(position_mask, dtype=tf.float32)
        training_data = position_mask * simulate_data

        bandwidth, kernel_shape = bandwidth_preparation_big(position_mask, Ch,
                                                            increase_kernel_size=increase_kernel_size)

        kem = KEM_SIMU_complex(K=3, shape=CT_shape,
                               training_data=training_data,
                               position_mask=position_mask,
                               kernel_shape=kernel_shape,
                               bandwidth=bandwidth,
                               kmeans_sample_ratio=1 / 100 / training_ratio,
                               testing_data=None)
        # KEM algorithm
        kem.kem_algorithm(max_steps=20, epsilon=1e-4, smooth_parameter=1e-20)
        # compute RMSE
        pi_rmse, mu_rmse, sigma_rmse = compute_RMSE(kem, pi, mu, sigma)
        print(f"\tUltimate MSE: pi_rmse: {pi_rmse:.4}; mu_rmse: {mu_rmse:.4}; sigma_rmse: {sigma_rmse:.4}")
        # record results
        mse_results = [f"patient_{index}", seed, CT_shape[0], training_ratio, kernel_shape[0], pi_rmse, mu_rmse,
                       sigma_rmse]
        with open(path, 'a', newline='', encoding='utf-8') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(mse_results)