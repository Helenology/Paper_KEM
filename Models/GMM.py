#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/22
# @Author  : Helenology
# @Site    :
# @File    : GMM.py
# @Software: PyCharm


import time
from sklearn.cluster import KMeans
import sys

sys.path.append('../')
from utils import *


class GMM:
    """
    The class for (classic) Gaussian Mixture Model.
    """
    def __init__(self, K, shape,
                 training_data,
                 position_mask,
                 kmeans_sample_ratio=1 / 100,
                 testing_data=None):
        """
        Initialization.
        :param K: The number of the classes.
            It is consistently set to 3 in my simulation and experiment.
        :param shape: The shape of the CT data.
        :param training_data: The training data to be used for parameter estimation.
            CT data = training_data + testing_data
            Here, we use position_mask to denote whether a position belongs to the training_data (=1) or not (=0).
            training_data = CT data * position_mask
            testing_data = CT data * (1 - position_mask)
        :param position_mask: If =1, the position is training data; otherwise =0, is not training data.
        :param kmeans_sample_ratio: (100*kmeans_sample_ratio)% data are used for a fast initialization.
        :param testing_data: The data used for computing the SPE (Square Prediction Error).
        """
        ##############################################################
        #                              Input                         #
        ##############################################################
        self.position_mask = position_mask  # indicator for training data
        self.K = K                          # the number of classes
        self.shape = shape                  # shape of the CT data
        self.training_data = training_data  # training_data = CT data * position_mask
        self.testing_data = testing_data    # testing_data = CT data * (1 - position_mask); or None if position_mask=1
        self.current_steps = 0              # current step in the optimization

        ##############################################################
        #                  Parameter Initialization                  #
        ##############################################################
        # pik: posterior probability
        self.pik_estimate = tf.ones((self.K, *shape, 1)) / self.K  # with shape (K, 1, 1, 1, 1)
        # pi: prior probability
        self.pi_estimate = tf.ones((self.K, 1, 1, 1, 1)) / self.K  # with shape (K, 1, 1, 1, 1)
        # sigma: standard deviation
        self.sigma_estimate = tf.ones((self.K, 1, 1, 1, 1)) * 0.05  # with shape (K, 1, 1, 1, 1)
        # mu: mean
        ###################### subsampling kmeans initialization for mu
        self.mu_estimate = tf.ones((self.K, 1, 1, 1, 1))
        print(f"From function(__init__): Initialize mu via kmeans(with K={self.K})")
        t1 = time.time()
        x = self.training_data[self.position_mask > 0.5]  # select available training data
        x = tf.reshape(x, [-1, 1])                        # reshape the data into a 2-dimensional array
        # randomly select (kmeans_sample_ratio) data
        random_x_sample_index = np.random.binomial(n=1, p=kmeans_sample_ratio, size=x.shape[0])
        random_x_sample = x[random_x_sample_index == 1]   # maintain the chosen data for kmeans
        print(f"From function(__init__): Randomly pick {random_x_sample_index.sum() / x.shape[0]:.4} data for kmeans.")
        model = KMeans(n_clusters=self.K)                 # kmeans model
        model.fit(random_x_sample)                        # kmeans fitting
        centers = model.cluster_centers_                  # kmeans centers
        centers = centers.reshape((self.K,))              # reshape the centers into vector form
        centers = sorted(centers, reverse=True)           # order the centers in descending order
        centers = tf.reshape(tf.cast(tf.constant(centers), tf.float32), [self.K, 1, 1, 1, 1])
        self.centers = centers
        self.mu_estimate *= self.centers                  # keep the kmeans centers as the mean initialization
        t2 = time.time()
        print(f"From function(__init__): KMeans(with K={self.K}) success, with time: {t2 - t1:.4} seconds\n"
              f"\tcenters: {tf.squeeze(self.centers)}")
        print("From function(__init__): Initialize parameters successfully.")
        print(f"\tpik_estimate:{self.pik_estimate.shape}\n\tpi_estimate: {self.pi_estimate.shape}\n"
              f"\tmu_estimate: {self.mu_estimate.shape}\n\tsigma_estimate: {self.sigma_estimate.shape}")

    def gmm_algorithm(self, max_steps, epsilon, smooth_parameter=1e-5):
        """
        GMM algorithm.
        :param max_steps: The max iteration steps.
        :param epsilon: The terminal condition of the distance of the estimators in two consecutive steps.
        :param smooth_parameter: The smoothing term to avoid python errors, e.g., 0/0.
        :return:
        """
        print(f"From function(gmm_algorithm): Receive max_steps: {max_steps}.")
        FLAG_CONTINUE = True  # used for terminal condition

        while FLAG_CONTINUE:
            t1 = time.time()
            print(f"########################## STEP {self.current_steps} ##########################")
            # In each iteration, run e step and m step sequentially
            self.e_step(smooth_parameter)
            print(f"From function(gmm_algorithm): E step success.")
            difference = self.m_step(smooth_parameter)
            print(f"From function(gmm_algorithm): M step success.")
            print(f"From function(gmm_algorithm): difference: {difference:.6}.")
            # terminal condition
            if self.current_steps >= max_steps or np.isnan(difference) or difference < epsilon:
                FLAG_CONTINUE = False
            self.current_steps += 1
            t2 = time.time()
            print(f"---This iteration step costs {t2 - t1:.4} seconds.---")

            # re-organize mu in descending order
            GMM_mu_estimate = tf.squeeze(self.mu_estimate).numpy()
            index_order = np.argsort(-GMM_mu_estimate.reshape(-1))  # in descending order
            if np.sum(index_order != np.arange(self.K)) > 0:
                print(f"From function(gmm_algorithm): Reorganize the order of the classes.")
                # reorganize the parameter estimators
                GMM_pi_estimate = tf.squeeze(self.pi_estimate).numpy()
                GMM_sigma_estimate = tf.squeeze(self.sigma_estimate).numpy()
                GMM_mu_estimate = [GMM_mu_estimate[index] for index in index_order]
                GMM_pi_estimate = [GMM_pi_estimate[index] for index in index_order]
                GMM_sigma_estimate = [GMM_sigma_estimate[index] for index in index_order]
                # save the parameter estimators
                self.mu_estimate = tf.reshape(tf.convert_to_tensor(GMM_mu_estimate, dtype=tf.float32),
                                              (self.K, 1, 1, 1, 1))
                self.pi_estimate = tf.reshape(tf.convert_to_tensor(GMM_pi_estimate, dtype=tf.float32),
                                              (self.K, 1, 1, 1, 1))
                self.sigma_estimate = tf.reshape(tf.convert_to_tensor(GMM_sigma_estimate, dtype=tf.float32),
                                                 (self.K, 1, 1, 1, 1))
                # reorganize the posterior prob
                old_pik = tf.squeeze(self.pik_estimate).numpy()
                pik_estimate = np.zeros_like(old_pik)
                for k in range(self.K):
                    pik_estimate[k] = old_pik[index_order[k]]
                # save the posterior prob
                self.pik_estimate = tf.reshape(tf.convert_to_tensor(pik_estimate, dtype=tf.float32),
                                               self.pik_estimate.shape)

    def e_step(self, smooth_parameter=1e-6):
        """
        The E step of the GMM algorithm.
        :param smooth_parameter: The smoothing term to avoid python errors, e.g., 0/0.
        :return:
        """
        # update the posterior probability
        pik_estimate = normal_density_function_tf((self.training_data - self.mu_estimate) / self.sigma_estimate) * (
                self.pi_estimate / self.sigma_estimate)
        pik_estimate_sum = tf.reduce_sum(pik_estimate, axis=0)
        # pik_estimate_sum cannot be 0 since it will serve as denominator
        if tf.reduce_sum(tf.cast(pik_estimate_sum == 0, tf.float32)) > 0:
            print(f"+++ From m_step: add smooth_parameter to pik_estimate")
            pik_estimate += smooth_parameter                        # add smooth parameter to denominator
            pik_estimate_sum = tf.reduce_sum(pik_estimate, axis=0)  # later adjust the sum to be 1
        pik_estimate /= pik_estimate_sum                            # adjust the sum to be 1
        pik_difference = tf.reduce_mean((self.pik_estimate - pik_estimate) ** 2)
        print(f"\t Current pik difference: {pik_difference:.6}")
        self.pik_estimate = pik_estimate                            # save the posterior estimators

    def m_step(self, smooth_parameter=1e-6):
        """
        The M step of the GMM algorithm.
        :param smooth_parameter: The smoothing term to avoid python errors.
        :return: The difference of estimators between two consecutive steps.
        """
        # update prior estimators (pi_estimate)
        pi_estimate = tf.reduce_sum(self.pik_estimate * self.position_mask, axis=(1, 2, 3), keepdims=True)
        # pi_estimate cannot be 0 since it will serve as denominator
        if tf.reduce_sum(tf.cast(pi_estimate == 0, tf.float32)) > 0:
            print(f"+++ From m_step: add smooth_parameter to pi_estimate")
            pi_estimate += smooth_parameter                # add smooth parameter
        # N = self.shape[0] * self.shape[1] * self.shape[2]
        N = tf.reduce_sum(self.position_mask)
        pi_estimate /= N
        print(f"pi_estimator: {tf.squeeze(pi_estimate)}")

        # update mean estimators (mu_estimate)
        mu_estimate = self.pik_estimate * self.training_data * self.position_mask
        mu_estimate = tf.reduce_sum(mu_estimate, axis=(1, 2, 3), keepdims=True)
        mu_estimate /= (pi_estimate * N)
        print(f"\nmu_estimate: {tf.reshape(self.mu_estimate, (-1,))}")

        # update standard deviation estimators (sigma_estimate)
        sigma_estimate = self.pik_estimate * (self.training_data - mu_estimate)**2 * self.position_mask
        sigma_estimate = tf.reduce_sum(sigma_estimate, axis=(1, 2, 3), keepdims=True)
        sigma_estimate /= (pi_estimate * N)
        sigma_estimate = tf.sqrt(sigma_estimate)
        print(f"sigma_estimator: {tf.squeeze(sigma_estimate)}")
        # sigma_estimate cannot be 0
        if tf.reduce_sum(tf.cast(sigma_estimate == 0, tf.float32)) > 0:
            print(f"From m_step: add smooth_parameter to sigma_estimate")  # add smooth parameter
            sigma_estimate += smooth_parameter

        # compute estimator differences
        pi_difference = tf.reduce_mean((self.pi_estimate - pi_estimate) ** 2)
        print(f"\t Current pi difference: {pi_difference:.6}")
        mu_difference = tf.reduce_mean((self.mu_estimate - mu_estimate) ** 2)
        print(f"\t Current mu difference: {mu_difference:.6}")
        sigma_difference = tf.reduce_mean((self.sigma_estimate - sigma_estimate) ** 2)
        print(f"\t Current sigma difference: {sigma_difference:.6}")
        difference = pi_difference + mu_difference + sigma_difference

        # save updated estimators
        self.pi_estimate = pi_estimate
        self.mu_estimate = mu_estimate
        self.sigma_estimate = sigma_estimate
        return difference.numpy()

    def compute_prediction_error(self):
        """
        Compute SPE (Square Prediction Error).
        :return: computed SPE metric
        """
        assert self.testing_data is not None
        # mask=1 then is testing data; otherwise should not be used
        testing_position_mask = 1 - self.position_mask
        testing_size = tf.reduce_sum(testing_position_mask)
        Y = tf.squeeze(self.testing_data)  # true Y
        # predict Y
        predict_Y = tf.reduce_sum(tf.squeeze(self.pi_estimate * self.mu_estimate * testing_position_mask), axis=0)
        # SPE
        prediction_error = tf.reduce_sum((Y - predict_Y) ** 2) / testing_size
        prediction_error = prediction_error.numpy()
        return prediction_error

    def predict_test_class(self):
        """
        Predict the classes for testing data.
        :return: predicted testing classes
        """
        assert self.testing_data is not None
        # compute the posterior probability of the testing data
        self.predict_prob = self.pi_estimate / self.sigma_estimate
        self.predict_prob *= normal_density_function_tf((self.testing_data - self.mu_estimate) / self.sigma_estimate)
        self.predict_prob /= tf.reduce_sum(self.predict_prob, axis=0)
        # class = argmax_k posterior probability
        self.all_predict_class = tf.cast(tf.argmax(self.predict_prob, axis=0), tf.float32)

        # select only the voxel positions of the testing positions
        self.predict_class = tf.reshape(self.all_predict_class, self.position_mask.shape)
        self.predict_class = self.predict_class[self.position_mask < 0.5]
        return self.predict_class
