import numpy as np
import scipy.io
import time
from collections import Counter
import math

import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.client import device_lib

from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

from parser import *
from helper import *
from result_helper import *
from hyper_params import *
from mhn import *


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    fn, fext = os.path.splitext(filename)
    start_time = time.time()
    
    if filename == 'fmnist.csv':
        # load fashion-mnist dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        M = x_train
        M = tf.reshape(M, shape=(M.shape[0], M.shape[1]*M.shape[2]))
        Y_true = y_train
        clcnt = len(set(Y_true))
    elif filename == 'Yale.mat':
        # for .mat/.sci files
        M, Y = read_file(filename, delimiter)
        (n, d) = np.shape(M)
        # for .mat file
        Y -= 1
        Y_true = Y.flatten()
        clcnt = len(set(Y_true))
    elif fext == ".npz":
        data = np.load('data/' + filename, allow_pickle=True)
        M = data['X']
        Y = data['Y']
        Y_true = np.array(list(map(int, Y)))
        Y_true -= 1
        clcnt = len(set(Y_true))
    else:
        D = read_file(filename, delimiter, label_filename)
        (n, d) = np.shape(D)  # get input dimensions
        d = d - 1
        M = D[:, 0:d]
        Y_true = D[:, d].astype(int) 
        clcnt = get_class_count()

    print("M", np.shape(M), "Y_true", np.shape(Y_true), 'clcnt', clcnt)
    print_distribution(Y_true.tolist(), clcnt)

    # StandardScaler
    scaler = StandardScaler()
    M = scaler.fit_transform(M)

    # set model params
    N1 = M.shape[1]
    N2 = clcnt
    input_shape = N1
    c = 1
    Ns = M.shape[0] # len of train_data

    dic = []
    mem_dic = []
    iter = 0
    printed_flag = False

    for initial_learning_rate_kernel in learning_rate_kernel_list:
        for beta in beta_list:
            for alpha in alpha_list:
                for batch_size in batch_size_list:
                    for mask_prob in mask_prob_list:
                        for N_steps_add in N_steps_add_list:
                            for dim_change_cycle in dim_change_cycle_list:
                                for mask_value in mask_value_list:
                                    iter += 1
                                    # get the exact mask_values
                                    if mask_value == '0':
                                        mask_values = np.zeros(N1)
                                    elif mask_value == 'min':
                                        mask_values = np.amin(M, axis=0)
                                    elif mask_value == 'max':
                                        mask_values = np.amax(M, axis=0)
                                    else:
                                        mask_values = np.mean(M, axis=0)
                                    for init in range(N_init):
                                        print('-----------iter:', iter, ' init:', init+1, '-----------')
                                        N_steps = int(1/alpha) + N_steps_add
                                        learning_rate_kernel = initial_learning_rate_kernel

                                        # define model
                                        input_mask = Input(shape=[input_shape])
                                        input1 = Input(shape=[input_shape])
                                        MHN_cell = MHN_WITH_1_HIDDEN_LAYER(N1, N2, beta, alpha)
                                        x = MHN_cell(input1, input_mask)
                                        for i in range(N_steps-1):
                                            x = MHN_cell(x, input_mask)

                                        model = Model(inputs=[input1, input_mask], outputs=x)
                                        
                                        # set optimizer
                                        optimizer = Adam(learning_rate=learning_rate_kernel)
                                        # create train and test dataset
                                        dataset = tf.data.Dataset.from_tensor_slices((M, M))
                                        train_dataset = dataset.shuffle(Ns).batch(batch_size)
                                        test_dataset = dataset.batch(batch_size)

                                        # set initial variables
                                        prev_mean_loss = 100000
                                        patience = 0
                                        optimum_lr = learning_rate_kernel
                                        optimum_epoch = 1
                                        optimum_memory = []
                                        losses_schedular = []
                                        losses = []
                                        epochs = []
                                        min_loss = 10000
                                        sum_losses = 0
                                        initial_loss = 0
                                        nan_flag = False
                                       
                                        # start training the model
                                        # track traininig time
                                        start_train_time = time.time()
                                        for epoch in range(1, N_ep+1):
                                            # print(f'Epoch {epoch}/{N_ep}')
                                            sum_loss = 0
                                            is_entire_dim = False
                                            if epoch % dim_change_cycle == 0:
                                                is_entire_dim = True

                                            for step, (x_train, y_train) in enumerate(train_dataset):
                                                x_train = tf.cast(x_train,dtype=tf.float32)
                                                y_train = tf.cast(y_train,dtype=tf.float32)
                                                
                                                # # without mask training
                                                # x_train_masked = x_train
                                                # mask = tf.cast(tf.equal(x_train, y_train), dtype=tf.float32)
                                                
                                                # with mask training
                                                x_train_masked, mask = construct_masked_data(x_train, N1, mask_prob, mask_values, is_entire_dim)
                                                
                                                KS = model.layers[2].get_weights()[0]

                                                # update weights
                                                with tf.GradientTape() as tape:
                                                    y_pred = model([x_train_masked, mask])
                                                    loss = mean_squared_loss(y_train, y_pred, mask)
                                                
                                                sum_loss += loss
                                                
                                                gradients = tape.gradient(loss, model.trainable_variables)
                                                optimizer.learning_rate.assign(learning_rate_kernel)  # for optimizing memories
                                                optimizer.apply_gradients(zip(gradients[:1], model.trainable_variables[:1]))

                                                for _ in range(1, len(model.trainable_variables)):
                                                    model.trainable_variables[_].assign(tf.clip_by_value(model.trainable_variables[_], clip_low, clip_high))

                                            if math.isnan(sum_loss):
                                                nan_flag = True
                                                print('Got nan value for beta', beta, ': Aborting the config..')
                                                break

                                            sum_losses += sum_loss
                                            losses_schedular.append(sum_loss)
                                            
                                            losses.append(sum_loss)
                                            epochs.append(epoch)
                                            
                                            if epoch == 1:
                                                initial_loss = sum_loss

                                            if sum_loss < min_loss:
                                                min_loss = sum_loss
                                                if not os.path.exists(model_directory):
                                                    os.makedirs(model_directory)
                                                model.save(model_directory + filename)
                                                optimum_lr = learning_rate_kernel
                                                optimum_epoch = epoch
                                                optimum_memory =  model.layers[2].get_weights()[0].tolist()
                                            
                                            if epoch >= 10:
                                                mean_loss = sum_losses/10
                                                sum_losses -= losses_schedular.pop(0)
                                                loss_diff = prev_mean_loss - mean_loss
                                                if loss_diff < loss_threshold:
                                                    patience += 1
                                                    if patience == max_patience:
                                                        learning_rate_kernel *= reduce_frac
                                                        patience = 0
                                                else: 
                                                    patience = 0

                                                prev_mean_loss = mean_loss

                                            if learning_rate_kernel < learning_rate_threshold:
                                                print('Learning Rate is too low! Breaking the loop!')
                                                break  

                                        if nan_flag:
                                            break

                                        end_train_time = time.time()
                                        print("Training: Took", (end_train_time - start_train_time), "seconds to complete one config for epoch", epoch, "for", filename, "dataset.")


                                        # inference step 
                                        # get the best model from saved directory in respect to minimum loss
                                        model = tf.keras.models.load_model(model_directory + filename, compile=False)  
                                        start_test_time = time.time()              
                                        label_pred = []
                                        for step, (x_train, y_train) in enumerate(test_dataset):
                                            x_train = tf.cast(x_train,dtype=tf.float32)
                                            y_train = tf.cast(y_train,dtype=tf.float32)
                                            
                                            mask = tf.cast(tf.equal(x_train, y_train), dtype=tf.float32)
                                            
                                            y_pred = model([x_train, mask])
                                            KS = model.layers[2].get_weights()[0]

                                            # for euclidian am
                                            KS = tf.squeeze(KS)
                                            label_pred += get_final_clusters(KS, y_pred)

                                        end_test_time = time.time()
                                        print("Inference: Took", (end_test_time - start_test_time), "seconds to complete one config for", filename, "dataset.")

                                        # # get baseline labels
                                        # if not printed_flag:
                                        #     kmeans_labels, spectral_labels, agglomerative_labels = get_basline_cluster_labels(M, N2)

                                        # # # print baseline results for testing purpose
                                        # # print('kmeans: number_of_cluster_found:', len(set(kmeans_labels)))
                                        # # print('Silhouette_score of kmeans_labels:', silhouette_score(M, kmeans_labels, metric='euclidean'))
                                        # # print('spectral_labels: number_of_cluster_found:', len(set(spectral_labels)))
                                        # # print('Silhouette_score of spectral_labels:', silhouette_score(M, spectral_labels, metric='euclidean'))
                                        # # print('agglomerative_labels: number_of_cluster_found:', len(set(agglomerative_labels)))
                                        # # print('Silhouette_score of agglomerative_labels:', silhouette_score(M, agglomerative_labels, metric='euclidean'))

                                        # print distributions
                                        # print_distributions_of_clusters(Y_true, label_pred, kmeans_labels, spectral_labels, agglomerative_labels, N2, printed_flag)

                                        # print cluster similarities matrices
                                        # print_cluster_similarities_matrices(Y_true, label_pred, kmeans_labels, spectral_labels, agglomerative_labels, printed_flag)
                                        # printed_flag = True
                                        
                                        sc_euclidean = -1
                                        num_of_cluster = len(Counter(label_pred).keys())
                                        if num_of_cluster != 1:
                                            sc_euclidean = silhouette_score(M, label_pred, metric='euclidean', sample_size=100000)

                                        # JSON object
                                        dictionary = {
                                            "iter": iter,
                                            "N_steps": N_steps,
                                            "beta": beta,
                                            "alpha": alpha,
                                            "batch_size": batch_size,
                                            "initial_learning_rate_kernel": initial_learning_rate_kernel,
                                            "learning_rate_kernel": learning_rate_kernel,
                                            "mask_prob": mask_prob,
                                            "mask_value": mask_value,
                                            "dim_change_cycle": dim_change_cycle,
                                            "initial_loss": str(initial_loss.numpy()),
                                            "min_loss": str(min_loss.numpy()),
                                            "optimum_lr": optimum_lr,
                                            "optimum_epoch": optimum_epoch,
                                            "init": init,
                                            "number_of_cluster_found": num_of_cluster,
                                            "training_time": end_train_time - start_train_time,
                                            "assignment_time": end_test_time - start_test_time,
                                            "num_of_cluster_after_adjustment": -1,
                                            "nmi_after_adjustment": -1,
                                            "ari_after_adjustment": -1,
                                            "sc_cosine_after_adjustment": -1,
                                            "sc_euclidean_after_adjustment": -1,
                                            "nmi": normalized_mutual_info_score(Y_true, label_pred),
                                            "ari": adjusted_rand_score(Y_true, label_pred),
                                            "sc_euclidean": sc_euclidean
                                        }
                                        dic.append(dictionary)


    # write dictionary to file
    sort_key = 'sc_euclidean'
    write_dic_to_file(dic, mem_dic, sort_key, directory, filename, suffix, ext)

    end_time = time.time()
    print("Took", (end_time - start_time)/3600, "hours to complete",  total_config,  "iterations with", N_init, "re-start for", filename, "dataset.")
