# from time import time
import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Input, Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import tensorflow as tf
from helper import *


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` which represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DCEC(object):
    def __init__(self,
                 input_shape,
                 n_clusters=10,
                 alpha=1.0):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        input = Input(shape=[input_shape[0]])
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')
        self.model = Model(inputs=input, outputs=clustering_layer(input))

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld'], loss_weights=[1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, init, y=None, batch_size=256, maxiter=2e4, tol=1e-5,
            update_interval=140, cae_weights=None, save_dir='./results/temp'):

        if init == 'kmeans':
            # Step 2: initialize cluster centers using k-means
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=kmeans_n_init_list[0])
            self.y_pred = kmeans.fit_predict(x)
            y_pred_last = np.copy(self.y_pred)
            self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        else:
            cluster_centers = np.random.normal(mu, sigma, size=(self.n_clusters, x.shape[1]))
            self.model.get_layer(name='clustering').set_weights([cluster_centers])

        loss = [0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                loss = np.round(loss, 5)
                print('Iter', ite, '; loss=', loss)

                if ite == 0:
                     y_pred_last = np.copy(self.y_pred)
                     continue

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::], y=p[index * batch_size::])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size], y=p[index * batch_size:(index + 1) * batch_size])
                index += 1

            ite += 1

        return self.model.get_layer(name='clustering').get_weights()


def DCEC_clustering(x, n_clusters, skmeans_update_interval, init):
    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], n_clusters=n_clusters)

    optimizer = 'adam'
    dcec.compile(loss=['kld'], loss_weights=[1], optimizer=optimizer)
    cluster_centers = dcec.fit(x, init, y=None, tol=skmeans_tolerance, maxiter=skmeans_maxiter, update_interval=skmeans_update_interval)
    y_pred = dcec.y_pred
    return y_pred, cluster_centers