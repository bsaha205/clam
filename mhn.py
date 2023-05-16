import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from hyper_params import *
import numpy as np

class MHN_WITH_1_HIDDEN_LAYER(tf.keras.layers.Layer):
    def __init__(self, N1, N2, beta, alpha, init_mem=None, c=1, **kwargs):
        super().__init__(**kwargs)
        self.N1 = N1
        self.N2 = N2
        self.c = c
        self.beta = beta
        self.alpha = alpha
        self.init_mem = init_mem
    
    def build(self, input_shape):
        # for original kmeans
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.N2, self.N1, 1],
            initializer=RandomNormal(mean=mu, stddev=sigma, seed=None) 
            # initializer=tf.constant_initializer(self.init_mem)
        )

        # for weighted clam
        # self.weight = self.add_weight(
        #     "weight",
        #     shape=[self.N2, 1], 
        #     initializer=RandomNormal(mean=mu, stddev=sigma, seed=None) 
        # )

        super().build(input_shape)

    def call(self, v, mask):
        Mem = self.kernel
        v = tf.expand_dims(v, axis=-1)
        v = tf.transpose(v, perm=[2, 1, 0])
        diff = Mem - v 
        
        # original clam
        exp_sum_diff = tf.exp((-self.beta/2)*tf.reduce_sum(diff**2, axis=1))
        den = tf.expand_dims(tf.reduce_sum(exp_sum_diff, axis=0),axis=0)
        num = tf.reduce_sum(diff*tf.expand_dims(exp_sum_diff,axis=1),axis=0) 

        ## for weighted clam
        # exp_sum_diff = tf.exp((-self.beta/2)*tf.reduce_sum(diff**2, axis=1))
        # num = tf.reduce_sum(diff*tf.expand_dims(self.weight*exp_sum_diff,axis=1),axis=0)
        # den = tf.expand_dims(tf.reduce_sum(self.weight*exp_sum_diff, axis=0),axis=0) 
     
        update = num/den 

        # with mask
        mask = tf.transpose(tf.expand_dims(mask, axis=0), perm=[0, 2, 1])
        v += self.alpha*tf.expand_dims(update, axis=0) * mask

        v = tf.transpose(v, perm=[2, 1, 0])
        v = tf.squeeze(v, axis=2)

        return v