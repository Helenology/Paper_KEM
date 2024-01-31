#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/22
# @Author  : Helenology
# @Site    :
# @File    : Kmeans.py
# @Software: PyCharm


import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sys

sys.path.append('../')
from utils import *


class Kmeans:
    """
    The class for (classic) Kmeans Model.
    """
    def __init__(self, K, shape,
                 training_data,
                 position_mask,
                 kmeans_sample_ratio=1 / 100,
                 testing_data=None):
        """
        Initialization.
        :param K: The number of the classes.
        :param shape: The shape of the CT data.
        :param training_data:
            The data used for estimate the parameters.
            Only the non-zero position has useful info of the CT data.
        :param position_mask: If =1, the voxel position is selected; otherwise =0, not selected.
        :param kmeans_sample_ratio: (100*?)% data used for kmeans initialization.
        :param testing_data: The data used for computing the SPE(square prediction error).
        """
        self.kmeans = None
        self.position_mask = position_mask
        self.K = K
        self.shape = shape
        self.training_data = training_data  # tf.reshape(training_data, (-1, 1))
        self.testing_data = testing_data
        self.mu_estimate = None
        self.pi_estimate = None
        self.sigma_estimate = np.zeros(self.K)

        # kmeans sampling initialization
        x = self.training_data[self.position_mask > 0.5]  # select available training data
        x = tf.reshape(x, [-1, 1])  # reshape the data into a 2-dimensional array
        # Randomly select (kmeans_sample_ratio) data to conduct the following kmeans algo
        random_x_sample_index = np.random.binomial(n=1, p=kmeans_sample_ratio, size=x.shape[0])
        random_x_sample = x[random_x_sample_index == 1]  # select the chosen data for kmeans
        print(f"From function(__init__): Randomly pick {random_x_sample_index.sum() / x.shape[0]:.4} data for kmeans.")
        model = KMeans(n_clusters=self.K, random_state=0)  # kmeans algorithm
        model.fit(random_x_sample)  # fit random sample from training data
        self.centers = -np.sort(-model.cluster_centers_.reshape(self.K,))  # kmeans centers
        self.centers = self.centers.reshape((self.K, 1))

    def kmeans_algorithm(self, max_steps):
        """
        KEM algorithm
        :param max_steps: The max iteration steps.
        :return:
        """
        training_data = self.training_data[self.position_mask > 0.5]
        training_data = training_data.numpy()
        training_data = training_data.reshape((-1, 1))
        self.kmeans = KMeans(n_clusters=self.K,
                             random_state=0,
                             max_iter=max_steps,
                             init=self.centers)
        self.kmeans.fit(training_data)

        # Compute the mean
        self.mu_estimate = self.kmeans.cluster_centers_.reshape(self.K, )
        self.mu_estimate = -np.sort(-self.mu_estimate)

        # Compute the std
        # re-organize the clustering orders
        index_order = np.argsort(-self.kmeans.cluster_centers_.reshape(self.K, ))  # in descending order
        tmp_labels = self.kmeans.labels_  # in the original order
        table = pd.Series(tmp_labels).value_counts() / len(tmp_labels)
        self.pi_estimate = np.array(table[index_order])

        for k in range(self.K):
            data_k = training_data[tmp_labels == index_order[k]] - self.mu_estimate[k]
            sigma2_k = np.mean(data_k**2)
            self.sigma_estimate[k] = np.sqrt(sigma2_k)

    def compute_prediction_error(self):
        """
        Compute SPE(square prediction error) on the testing data.
        :return:
        """
        assert self.testing_data is not None
        pi_estimate = tf.cast(tf.reshape(self.pi_estimate, (self.K, 1, 1, 1, 1)), tf.float32)
        mu_estimate = tf.cast(tf.reshape(self.mu_estimate, (self.K, 1, 1, 1, 1)), tf.float32)
        # mask=1 then is testing data; otherwise should not be used
        testing_position_mask = 1 - self.position_mask
        testing_size = tf.reduce_sum(testing_position_mask)
        Y = tf.squeeze(self.testing_data)  # [depth, height, width]
        # predict Y
        predict_Y = tf.reduce_sum(tf.squeeze(pi_estimate * mu_estimate * testing_position_mask), axis=0)
        predict_Y = tf.squeeze(predict_Y)

        # the SPE
        prediction_error = tf.reduce_sum((Y - predict_Y) ** 2) / testing_size
        prediction_error = prediction_error.numpy()
        return prediction_error

    def predict_test_class(self):
        testing_data = tf.reshape(self.testing_data[self.position_mask < 0.5], [-1, 1])
        testing_data = testing_data.numpy()

        # re-organize the clustering orders
        index_order = np.argsort(-self.kmeans.cluster_centers_.reshape(self.K, ))  # in descending order
        tmp_labels = self.kmeans.predict(testing_data)  # in the original order
        labels = np.zeros_like(tmp_labels)

        # re-order the clustering label
        for k in range(self.K):
            labels[tmp_labels == index_order[k]] = k

        # compute prediction accuracy
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        return labels

    def predict_all_class(self):
        data = tf.reshape(self.training_data, [-1, 1])
        data = data.numpy()
        print(self.training_data.shape)

        # re-organize the clustering orders
        index_order = np.argsort(-self.kmeans.cluster_centers_.reshape(self.K, ))  # in descending order
        tmp_labels = self.kmeans.predict(data)  # in the original order
        labels = np.zeros_like(tmp_labels)

        # re-order the clustering label
        for k in range(self.K):
            labels[tmp_labels == index_order[k]] = k

        # compute prediction accuracy
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        labels = tf.reshape(labels, self.training_data.shape)
        return labels

