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
        self.position_mask = position_mask  # indicator for training data positions
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
        self.mu_estimate = tf.ones((self.K, *shape, 1))
        print(f"From function(__init__): Initialize mu via kmeans(with K={self.K})")
        t1 = time.time()
        x = self.training_data[self.position_mask > 0.5]   # select available training data
        x = tf.reshape(x, [-1, 1])                         # reshape the data into a 2-dimensional array
        # randomly select (kmeans_sample_ratio) data
        random_x_sample_index = np.random.binomial(n=1, p=kmeans_sample_ratio, size=x.shape[0])
        random_x_sample = x[random_x_sample_index == 1]    # maintain the chosen data for kmeans
        print(f"From function(__init__): Randomly pick {random_x_sample_index.sum() / x.shape[0]:.4} data for kmeans.")
        model = KMeans(n_clusters=self.K)                  # kmeans algorithm
        model.fit(random_x_sample)                         # kmeans fitting
        centers = model.cluster_centers_                   # kmeans centers
        centers = centers.reshape((self.K,))               # reshape the centers into vector form
        centers = sorted(centers, reverse=True)            # order the centers in descending order
        centers = tf.reshape(tf.cast(tf.constant(centers), tf.float32), [self.K, 1, 1, 1, 1])
        self.centers = centers
        self.mu_estimate *= self.centers                   # save the kmeans centers as the mean initialization
        t2 = time.time()
        print(f"From function(__init__): KMeans(with K={self.K}) success, with time: {t2 - t1:.4} seconds\n"
              f"\tcenters: {tf.squeeze(self.centers)}")
        print("From function(__init__): Initialize parameters successfully.")
        print(f"\tpik_estimate:{self.pik_estimate.shape}\n\tpi_estimate: {self.pi_estimate.shape}\n"
              f"\tmu_estimate: {self.mu_estimate.shape}\n\tsigma_estimate: {self.sigma_estimate.shape}")
        # kernel initialization
        self.kernel = self.generate_kernel(kernel_shape, bandwidth)  # generate the kernel/filter for 3D convolution
        print("From function(__init__): Initialize kernel successfully.")
        print(f"\tkernel: {self.kernel.shape}")

    def kem_algorithm(self, max_steps, epsilon, smooth_parameter=1e-5):
        """
        KEM algorithm
        :param max_steps: The max iteration steps
        :param epsilon: The terminal condition of the distance of the estimators in two consecutive steps.
        :param smooth_parameter: The smoothing term to avoid python errors, e.g., 0/0.
        :return:
        """
        print(f"From function(kem_algorithm): Receive max_steps: {max_steps}.")
        FLAG_CONTINUE = True  # used for terminal condition
        # the denominator of equation (2.6)
        self.denominator = tf.nn.conv3d(self.position_mask, self.kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        assert tf.reduce_sum(tf.cast(self.denominator == 0, dtype=tf.float32)) == 0  # denominator should not be zero

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
        :param smooth_parameter: The smoothing term to avoid python errors, e.g., 0/0.
        :return:
        """
        # update the posterior probability: $p_ik$ in (2.5)
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

    def m_step(self, smooth_parameter):
        """
        The M step of the KEM algorithm
        :param smooth_parameter: The smoothing term to avoid python errors.
        :return: The difference of estimators between two consecutive steps.
        """
        # update (the numerator of) the prior estimators via (2.6)
        pi_estimate = tf.nn.conv3d(self.pik_estimate * self.position_mask, self.kernel, strides=[1, 1, 1, 1, 1],
                                   padding="SAME")  # the numerator in (2.6)
        if tf.reduce_sum(tf.cast(pi_estimate == 0, tf.float32)) > 0:  # pi_estimate cannot be 0 as the denominator in (2.8)
            print(f"+++ From m_step: add smooth_parameter to pi_estimate")
            pi_estimate += smooth_parameter  # add smooth parameter

        # update the mean estimators via (2.7)
        mu_estimate = tf.nn.conv3d(self.pik_estimate * self.training_data * self.position_mask, self.kernel,
                                   strides=[1, 1, 1, 1, 1], padding="SAME")  # the numerator of $\mu$
        mu_estimate /= pi_estimate  # mean estimator in (2.7)

        # update the variance estimator via (2.8)
        sigma2_estimate = tf.nn.conv3d(self.pik_estimate * self.position_mask * (self.training_data - mu_estimate) ** 2,
                                       self.kernel, strides=[1, 1, 1, 1, 1], padding="SAME")  # the numerator in (2.8)
        sigma_estimate = tf.sqrt(sigma2_estimate / pi_estimate)  # std estimator in (2.8)
        if tf.reduce_sum(tf.cast(sigma_estimate == 0, tf.float32)) > 0:  # std cannot be 0
            print(f"From m_step: add smooth_parameter to sigma_estimate")
            sigma_estimate += smooth_parameter  # add smooth parameter

        # update  the prior estimators via (2.6)
        pi_estimate /= self.denominator  # prior estimator in (2.6)
        pi_estimate /= tf.reduce_sum(pi_estimate, axis=0)  # in case of accidents, normalize the pi again

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

    def generate_kernel(self, kernel_shape, bandwidth):
        """
        Generate the kernel for 3D convolution operations.
        :param kernel_shape: the shape of the kernel/filter
        :param bandwidth: the bandwidth used for the kernel weight
        :return:
        """
        kernel = np.zeros(kernel_shape)  # filter
        center = int((kernel_shape[0]) / 2)  # the center point's index
        center_coordinate = (center, center, center)  # the center point's coordinate
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
        Compute SPE (Square Prediction Error) on the testing data.
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
        # compute the posterior probability of the testing data via (2.5)
        self.predict_prob = self.pi_estimate / self.sigma_estimate
        self.predict_prob *= normal_density_function_tf((self.testing_data - self.mu_estimate) / self.sigma_estimate)
        self.predict_prob /= tf.reduce_sum(self.predict_prob, axis=0)
        # Class = argmax_k posterior probability
        self.all_predict_class = tf.cast(tf.argmax(self.predict_prob, axis=0), tf.float32)

        # select only the voxel positions of the testing positions
        self.predict_class = tf.reshape(self.all_predict_class, self.position_mask.shape)
        self.predict_class = self.predict_class[self.position_mask < 0.5]
        return self.predict_class