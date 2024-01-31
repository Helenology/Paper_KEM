import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
import sys
from tensorflow.keras.utils import to_categorical

sys.path.append('../')
from utils import *


class KEM_SIMU_complex():
    """
    the class for KEM algorithm in simulation
    """

    def __init__(self, K, shape,
                 training_data,
                 position_mask,
                 kernel_shape,
                 bandwidth=10 / 512,
                 kmeans_sample_ratio=1 / 100,
                 testing_data=None):
        """
        Initialize the algorithm
        :param K: which should be 3
        :param shape: the shape of the CT data
        :param training_data: if =0, the voxel position is not selected for KEM algorithm
        :param position_mask: if =1 the voxel position is selected for KEM algorithm; otherwise =0, not selected
        :param kernel_shape: the shape of the filter/kernel in the 3D convolution
        :param bandwidth: the bandwidth used in kernel smoothing
        :param kmeans_sample_ratio: ?% data used for kmeans
        :param testing_data: for computing SPE(square prediction error)
        """
        self.position_mask = position_mask
        self.K = K
        self.shape = shape
        self.training_data = training_data
        self.testing_data = testing_data
        self.current_steps = 0  # KEM algorithm iteration steps
        # Initialization
        self.pik_estimate = tf.ones((self.K, *shape, 1)) / self.K  # (K, depth, height, width, 1)
        self.pi_estimate = tf.ones((self.K, *shape, 1)) / self.K  # equal prior probability initialization
        self.sigma_estimate = tf.ones((self.K, *shape, 1)) * 0.05  # standard deviation initialization
        # mu initialization: using kmeans
        self.mu_estimate = tf.ones((self.K, *shape, 1))
        print(f"From function(__init__): Initialize mu via kmeans(with K={self.K})")
        t1 = time.time()
        x = self.training_data[self.position_mask > 0.5]  # select available data
        x = tf.reshape(x, [-1, 1])  # reshape the data into a 2-dimensional array
        random_x_sample_index = np.random.binomial(n=1, p=kmeans_sample_ratio, size=x.shape[0])
        random_x_sample = x[random_x_sample_index == 1]  # select the chosen data for kmeans
        print(
            f"From function(__init__): Randomly pick {random_x_sample_index.sum() / x.shape[0]:.4} positions for kmeans.")
        model = KMeans(n_clusters=self.K)  # kmeans algorithm
        model.fit(random_x_sample)
        centers = model.cluster_centers_
        centers = centers.reshape((3,))
        centers = sorted(centers, reverse=True)  # order the centers in descending order
        centers = tf.reshape(tf.cast(tf.constant(centers), tf.float32), [self.K, 1, 1, 1, 1])
        self.centers = centers
        self.mu_estimate *= self.centers
        t2 = time.time()
        print(
            f"From function(__init__): KMeans(with K={self.K}) success, with time: {t2 - t1:.4} seconds\n\tcenters: {tf.squeeze(self.centers)}")
        print("From function(__init__): Initialize parameters successfully.")
        print(
            f"\tpik_estimate:{self.pik_estimate.shape}\n\tpi_estimate: {self.pi_estimate.shape}\n\tmu_estimate: {self.mu_estimate.shape}\n\tsigma_estimate: {self.sigma_estimate.shape}")
        # kernel initialization
        self.kernel = self.generate_kernel(kernel_shape, bandwidth)  # generate the kernel/filter for 3D convolution
        print("From function(__init__): Initialize kernel successfully.")
        print(f"\tkernel: {self.kernel.shape}")

    def kem_algorithm(self, max_steps, epsilon, smooth_parameter=1e-5):
        """
        KEM algorithm
        :param max_steps: the max iteration steps
        :param epsilon: parameter gap
        :param smooth_parameter: smooth term
        :return:
        """
        print(f"From function(kem_algorithm): Receive max_steps: {max_steps}.")
        FLAG_CONTINUE = True  # used for abortion
        # the denominator of equation (2.9) of the article
        self.denominator = tf.nn.conv3d(self.position_mask, self.kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        # denominator should not be zero
        assert tf.reduce_sum(tf.cast(self.denominator == 0, dtype=tf.float32)) == 0

        while FLAG_CONTINUE:
            t1 = time.time()
            print(f"########################## STEP {self.current_steps} ##########################")
            # In each iteration, run e step and m step sequentially
            self.e_step(smooth_parameter)
            print(f"From function(kem_algorithm): E step success.")
            difference = self.m_step(smooth_parameter)
            print(f"From function(kem_algorithm): M step success.")
            print(f"From function(kem_algorithm): difference: {difference:.6}.")
            # end condition
            if self.current_steps >= max_steps or np.isnan(difference) or difference < epsilon:
                FLAG_CONTINUE = False

            self.current_steps += 1
            t2 = time.time()
            print(f"---This iteration step costs {t2 - t1:.4} seconds.---")

    def e_step(self, smooth_parameter):
        """
        The E step of the KEM algorithm
        :param smooth_parameter: a small term
        :return: nothing
        """
        # update parameters in the E step
        # $p_ik$ by equation (2.8) in the article
        pik_estimate = normal_density_function_tf((self.training_data - self.mu_estimate) / self.sigma_estimate) * (
                self.pi_estimate / self.sigma_estimate)
        pik_estimate_sum = tf.reduce_sum(pik_estimate, axis=0)
        # pik_estimate_sum cannot be 0 since it will serve as denominator
        if tf.reduce_sum(tf.cast(pik_estimate_sum == 0, tf.float32)) > 0:
            print(f"+++ From m_step: add smooth_parameter to pik_estimate")
            pik_estimate += smooth_parameter
            pik_estimate_sum = tf.reduce_sum(pik_estimate, axis=0)
        pik_estimate /= pik_estimate_sum  # sum should be 1
        pik_difference = tf.reduce_mean((self.pik_estimate - pik_estimate) ** 2)
        print(f"\t Current pik difference: {pik_difference:.6}")
        self.pik_estimate = pik_estimate  # update self.pik_estimate

    def m_step(self, smooth_parameter):
        """
        The M step of the KEM algorithm
        :param smooth_parameter: a small term
        :return: difference of estimators through updating
        """
        # update parameters in the M step
        # the numerator of $\pi$ by equation (2.9)
        pi_estimate = tf.nn.conv3d(self.pik_estimate * self.position_mask, self.kernel, strides=[1, 1, 1, 1, 1],
                                   padding="SAME")
        # pi_estimate cannot be 0 since it will serve as denominator in equations (2.10) and (2.11)
        if tf.reduce_sum(tf.cast(pi_estimate == 0, tf.float32)) > 0:
            print(f"+++ From m_step: add smooth_parameter to pi_estimate")
            pi_estimate += smooth_parameter
        # the numerator of $\mu$ by equation (2.10)
        mu_estimate = tf.nn.conv3d(self.pik_estimate * self.training_data * self.position_mask, self.kernel,
                                   strides=[1, 1, 1, 1, 1], padding="SAME")
        # $\mu$ by equation (2.10)
        mu_estimate /= pi_estimate
        # the numerator of $\sigmma^2$ by equation (2.11)
        sigma2_estimate = tf.nn.conv3d(self.pik_estimate * self.position_mask * (self.training_data - mu_estimate) ** 2,
                                       self.kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        sigma_estimate = tf.sqrt(sigma2_estimate / pi_estimate)
        # sigma_estimate cannot be 0 since it will serve as denominator in the E step
        if tf.reduce_sum(tf.cast(sigma_estimate == 0, tf.float32)) > 0:
            print(f"From m_step: add smooth_parameter to sigma_estimate")
            sigma_estimate += smooth_parameter
        # $\pi$ by equation (2.9)
        pi_estimate /= self.denominator
        pi_estimate /= tf.reduce_sum(pi_estimate, axis=0)  # in case of accidents, normalize the pi again

        # compute difference via updating
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
        Generate the kernel for 3D convolution operations.
        :param kernel_shape: the shape of the kernel/filter
        :param bandwidth: the bandwidth used for the kernel weight
        :return: nothing
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
        # reshape kernel as [filter_depth, filter_height, filter_width, in_channels,out_channels]
        kernel = tf.reshape(kernel, shape=(*kernel_shape, 1, 1))
        # kernel normalization
        #         kernel /= tf.reduce_sum(kernel)
        return kernel

    def compute_prediction_error(self):
        """
        compute SPE(square prediction error)
        :return:
        """
        assert self.testing_data is not None
        # mask=1 then is testing data; otherwise should not be used
        testing_position_mask = 1 - self.position_mask
        testing_size = tf.reduce_sum(testing_position_mask)
        Y = tf.squeeze(self.testing_data)  # [depth, height, width]
        # predict Y
        predict_Y = tf.reduce_sum(tf.squeeze(self.pi_estimate * self.mu_estimate * testing_position_mask), axis=0)
        # the SPE
        prediction_error = tf.reduce_sum((Y - predict_Y) ** 2) / testing_size
        prediction_error = prediction_error.numpy()
        return prediction_error

    def predict_test_class(self):
        assert self.testing_data is not None
        # Compute the posterior probability of the testing data
        self.predict_prob = self.pi_estimate / self.sigma_estimate
        self.predict_prob *= normal_density_function_tf((self.testing_data - self.mu_estimate) / self.sigma_estimate)
        self.predict_prob /= tf.reduce_sum(self.predict_prob, axis=0)
        # Class = argmax_k posterior probability
        self.all_predict_class = tf.cast(tf.argmax(self.predict_prob, axis=0), tf.float32)
        # Select only the voxel positions of the testing positions
        self.predict_class = tf.reshape(self.all_predict_class, self.position_mask.shape)
        self.predict_class = self.predict_class[self.position_mask < 0.5]

        return self.predict_class