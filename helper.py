import numpy as np
import tensorflow as tf
import json
from scipy.special import logsumexp
from hyper_params import *
import os


# for tabular data
def mean_squared_loss(y_true, y_pred, mask):
    loss = tf.reduce_mean(tf.square(y_true-y_pred)*mask)
    return loss

def energy_loss(v, memories, beta):
    v = tf.expand_dims(v, axis=-1)
    v = tf.transpose(v, perm=[2, 1, 0])
    loss = -logsumexp(-beta*tf.square(memories-v))/beta
    # print('energy_loss:', loss)
    return loss

def construct_masked_data(original_data, dimension, mask_prob, mask_value, is_entire_dim):
        masked_sequences = []
        masks = []
        
        original_data = original_data.numpy()
       
        if is_entire_dim:
            mask = np.zeros(dimension)
            mask_indxs = np.random.randint(1, dimension, (int(dimension*mask_prob)))
            mask[mask_indxs] = 1
            
            for s in original_data:
                masks.append(mask)

                s[mask_indxs] = mask_value[mask_indxs]
                masked_sequences.append(s.tolist())
        
        else:
            for s in original_data:
                mask = np.zeros(dimension)
                mask_indxs = np.random.randint(1, dimension, (int(dimension*mask_prob)))
                mask[mask_indxs] = 1
                masks.append(mask)

                s[mask_indxs] = mask_value[mask_indxs]
                masked_sequences.append(s.tolist())
        
        return tf.convert_to_tensor(masked_sequences, dtype=tf.float32), tf.convert_to_tensor(masks, dtype=tf.float32)

def make_configs_as_dic():
    dictionary = {
        "mu": mu,
        "sigma": sigma,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "beta_list": beta_list,
        "alpha_list": alpha_list,
        "N_steps_add_list": N_steps_add_list,
        "batch_size_list": batch_size_list,
        "learning_rate_kernel_list": learning_rate_kernel_list,
        "learning_rate_threshold": learning_rate_threshold,
        "N_ep": N_ep,
        "N_init": N_init,
        "max_patience": max_patience,
        'loss_threshold': loss_threshold,
        "reduce_frac": reduce_frac,
        "mask_prob_list": mask_prob_list,
        "mask_value_list": mask_value_list,
        "dim_change_cycle_list": dim_change_cycle_list,
        'kmeans_random_state_list': kmeans_random_state_list,
        "kmeans_n_init_list": kmeans_n_init_list,
        "spectral_n_init_list": spectral_n_init_list,
        'spectral_gamma_list': spectral_gamma_list,
        "spectral_affinity_list": spectral_affinity_list,
        "spectral_n_neighbors_list": spectral_n_neighbors_list,
        "spectral_assign_labels_list": spectral_assign_labels_list,
        "agglomerative_affinity_list": agglomerative_affinity_list,
        'agglomerative_linkage_list': agglomerative_linkage_list,
    }
    return dictionary


def write_dic_to_file(dic, mem_dic, sort_key, directory, filename, suffix, ext):
    #sort dic
    dic = sorted(dic, key=lambda k: k[sort_key], reverse=True)

    #convert list to dic
    dic_obj = {i+1: dic[i] for i in range(0, len(dic))}

    # Serializing json
    dic = {}
    dic['i_config'] = make_configs_as_dic()
    dic['results'] = dic_obj
    if store_memory:
        dic['memories'] = mem_dic
    json_object = json.dumps(dic, indent=4)

    # Writing to data.json
    directory = directory + filename + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + suffix + ext, "w") as outfile:
        outfile.write(json_object) 