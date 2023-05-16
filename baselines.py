import numpy as np
import scipy.io
import time
from collections import Counter
import math
import json

from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from parser import *
from helper import *
from result_helper import *
from draw_helper import *
from hyper_params import *
from mhn import *
import time
from s_kmeans import *


if __name__ == '__main__':
    start_time = time.time()
    fn, fext = os.path.splitext(filename)
    
    if filename == 'fmnist.csv':
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
        Y_true = Y.flatten()
        clcnt = len(set(Y_true))
    else:
        # read data file
        D = read_file(filename, delimiter, label_filename)
        (n, d) = np.shape(D)  # get input dimensions
        d = d - 1
        M = D[:, 0:d]
        Y_true = D[:, d].astype(int) 
        clcnt = get_class_count()

    print("M", np.shape(M), "Y_true", np.shape(Y_true), 'clcnt', clcnt)

    # StandardScaler
    scaler = StandardScaler()
    M = scaler.fit_transform(M)

    for baseline in baseline_list:
        dic = []
        iter = 0
        if baseline == 'kmeans':
            sort_key = 'silhouette_score_kemans_euclidean'
            for kmeans_n_init in kmeans_n_init_list:
                for kmeans_random_state in kmeans_random_state_list:
                    print('baseline:', baseline, ' iter:', iter)
                    
                    s_time = time.time()
                    kmeans = KMeans(n_clusters=clcnt, init='k-means++', random_state=kmeans_random_state, n_init=kmeans_n_init, max_iter=1000).fit(M)
                    kmeans_labels = kmeans.labels_.tolist()
                    
                    e_time = time.time()
                    print("Took", (e_time - s_time), "seconds to complete kmeans with", kmeans_n_init, "init")

                    scores = {
                        "iter": iter,
                        "kemans_n_init": kmeans_n_init,
                        "kemans_random_state": kmeans_random_state,
                        "number_of_cluster_found": len(set(kmeans_labels)),
                        "needed_time": (e_time - s_time),
                        "nmi_score_kemans": normalized_mutual_info_score(Y_true, kmeans_labels),
                        "ari_score_kemans": adjusted_rand_score(Y_true, kmeans_labels),
                        "silhouette_score_kemans_mahalanobis": -1,
                        "silhouette_score_kemans_cosine": -1,
                        "silhouette_score_kemans_euclidean": -1
                    }

                    # compute cluster quality matrices
                    try:
                        silhouette_score_kemans_euclidean = silhouette_score(M, kmeans_labels, metric='euclidean')
                        scores['silhouette_score_kemans_euclidean'] = silhouette_score_kemans_euclidean
                    except (ValueError):
                        print("Kmeans:: Oops! Only one cluster found!") 

                    dic.append(scores)
                    iter += 1
        
        elif baseline == 'nkmeans':
            sort_key = 'silhouette_score_kemans_euclidean'
            for kmeans_n_init in kmeans_n_init_list:
                for kmeans_random_state in kmeans_random_state_list:
                    for dataset in dataset_list:
                        for std in std_list:
                            print('baseline:', baseline, ' iter:', iter)
                        
                            if dataset == 'noisy':
                                M += np.random.normal(loc=0, scale=std, size=M.shape)
                            
                            s_time = time.time()
                            kmeans = KMeans(n_clusters=clcnt, init='k-means++', random_state=kmeans_random_state, n_init=kmeans_n_init).fit(M)
                            kmeans_labels = kmeans.labels_.tolist()
                            
                            e_time = time.time()
                            print("Took", (e_time - s_time), "seconds to complete kmeans with", kmeans_n_init, "init")

                            scores = {
                                "iter": iter,
                                "kemans_n_init": kmeans_n_init,
                                "kemans_random_state": kmeans_random_state,
                                "dataset": dataset,
                                "std": -1,
                                "number_of_cluster_found": len(set(kmeans_labels)),
                                "needed_time": (e_time - s_time),
                                "nmi_score_kemans": normalized_mutual_info_score(Y_true, kmeans_labels),
                                "ari_score_kemans": adjusted_rand_score(Y_true, kmeans_labels),
                                "nmi_score_with_original": -1,
                                "ari_score_with_original": -1,
                                "silhouette_score_kemans_euclidean": -1
                            }

                            # compute cluster quality matrices
                            try:
                                silhouette_score_kemans_euclidean = silhouette_score(M, kmeans_labels, metric='euclidean')
                                scores['silhouette_score_kemans_euclidean'] = silhouette_score_kemans_euclidean
                            except (ValueError):
                                print("Kmeans:: Oops! Only one cluster found!") 

                            if dataset == 'noisy':
                                scores['nmi_score_with_original'] = normalized_mutual_info_score(Y_original, kmeans_labels)
                                scores['ari_score_with_original'] = adjusted_rand_score(Y_original, kmeans_labels)
                                scores['std'] = std
                            else:
                                Y_original = kmeans_labels
                                dic.append(scores)
                                iter += 1
                                break

                            dic.append(scores)
                            iter += 1
        
        elif baseline == 'skmeans':
            sort_key = 'silhouette_score_skemans'
            for skmeans_update_interval in skmeans_update_interval_list:
                for init in init_list:
                    
                    s_time = time.time()

                    # skmeans clustering
                    skmeans_labels, cluster_centers = DCEC_clustering(M, clcnt, skmeans_update_interval, init)
                    
                    e_time = time.time()
                    print("Took", (e_time - s_time), "seconds to complete skmeans with", skmeans_update_interval, "skmeans_update_interval")
                    
                    scores = {
                        "iter": iter,
                        "skmeans_maxiter": skmeans_maxiter,
                        "skmeans_update_interval": skmeans_update_interval,
                        "init": init,
                        "number_of_cluster_found": len(set(skmeans_labels)),
                        "needed_time": (e_time - s_time),
                        "nmi_score_skemans": normalized_mutual_info_score(Y_true, skmeans_labels),
                        "ari_score_skemans": adjusted_rand_score(Y_true, skmeans_labels),
                        "silhouette_score_skemans": -1
                    }

                    # print cluster quality matrices
                    try:
                        silhouette_score_skemans = silhouette_score(M, skmeans_labels, metric='euclidean')
                        scores['silhouette_score_skemans'] = silhouette_score_skemans
                    except (ValueError):
                        print("Kmeans:: Oops! Only one cluster found!") 

                    print('baseline:', baseline, ' iter:', iter, ' skmeans_update_interval:', skmeans_update_interval, ' init:', init, ' silhouette_score_skemans:', silhouette_score_skemans)
                    
                    dic.append(scores)
                    iter += 1    
        
        elif baseline == 'spectral':
            sort_key = 'silhouette_score_spectral_euclidean'
            for spectral_affinity in spectral_affinity_list:
                for spectral_gamma in spectral_gamma_list:
                    for spectral_n_neighbors in spectral_n_neighbors_list:
                        for spectral_assign_labels in spectral_assign_labels_list:
                            for spectral_n_init in spectral_n_init_list:
                                print('baseline:', baseline, ' iter:', iter)
                                s_time = time.time()

                                # Spectral clustering
                                spectral = SpectralClustering(n_clusters=clcnt, n_init=spectral_n_init, gamma=spectral_gamma, affinity=spectral_affinity, n_neighbors=spectral_n_neighbors, assign_labels=spectral_assign_labels, random_state=0).fit(M)
                                spectral_labels = spectral.labels_.tolist()

                                e_time = time.time()
                                print("Took", (e_time - s_time), "seconds to complete spectral with", spectral_n_init, "init and affinity", spectral_affinity)

                                scores = {
                                    "iter": iter,
                                    "spectral_n_init": spectral_n_init,
                                    "spectral_gamma": spectral_gamma,
                                    "spectral_affinity": spectral_affinity,
                                    "spectral_n_neighbors": spectral_n_neighbors,
                                    "spectral_assign_labels": spectral_assign_labels,
                                    "number_of_cluster_found": len(set(spectral_labels)),
                                    "needed_time": (e_time - s_time),
                                    "nmi_score_spectral": normalized_mutual_info_score(Y_true, spectral_labels),
                                    "ari_score_spectral": adjusted_rand_score(Y_true, spectral_labels),
                                    "silhouette_score_spectral_mahalanobis": -1,
                                    "silhouette_score_spectral_cosine": -1,
                                    "silhouette_score_spectral_euclidean": -1
                                }

                                # compute cluster quality matrices
                                try:
                                    silhouette_score_spectral_cosine = silhouette_score(M, spectral_labels, metric='cosine')
                                    scores['silhouette_score_spectral_cosine'] = silhouette_score_spectral_cosine
                                    silhouette_score_spectral_euclidean = silhouette_score(M, spectral_labels, metric='euclidean')
                                    scores['silhouette_score_spectral_euclidean'] = silhouette_score_spectral_euclidean
                                except (ValueError):
                                    print("Spectral:: Oops! Only one cluster found!")   
                                except:
                                    print("Found numpy.linalg.LinAlgError: Singular matrix")

                                dic.append(scores)
                                iter += 1
                        if spectral_affinity == 'rbf':
                                    break
                    if spectral_affinity == 'nearest_neighbors':
                                    break

        elif baseline == 'agglomerative':
            sort_key = 'silhouette_score_agglomerative_euclidean'
            for agglomerative_affinity in agglomerative_affinity_list:
                for agglomerative_linkage in agglomerative_linkage_list:
                    if agglomerative_affinity != 'euclidean' and agglomerative_linkage == 'ward':
                        continue
                    print('baseline:', baseline, ' iter:', iter)
                    s_time = time.time()

                    # Agglomerative clustering
                    agglomerative = AgglomerativeClustering(n_clusters=clcnt, affinity=agglomerative_affinity, linkage=agglomerative_linkage).fit(M)
                    agglomerative_labels = agglomerative.labels_.tolist()

                    e_time = time.time()
                    print("Took", (e_time - s_time), "seconds to complete agglomerative with affinity", agglomerative_affinity, "and linkage", agglomerative_linkage)

                    scores = {
                        "iter": iter,
                        "agglomerative_affinity": agglomerative_affinity,
                        "agglomerative_linkage": agglomerative_linkage,
                        "number_of_cluster_found": len(set(agglomerative_labels)),
                        "needed_time": (e_time - s_time),
                        "nmi_score_agglomerative": normalized_mutual_info_score(Y_true, agglomerative_labels),
                        "ari_score_agglomerative": adjusted_rand_score(Y_true, agglomerative_labels),
                        "silhouette_score_agglomerative_mahalanobis": -1,
                        "silhouette_score_agglomerative_cosine": -1,
                        "silhouette_score_agglomerative_euclidean": -1
                    }

                    # compute cluster quality matrices
                    try:
                        silhouette_score_agglomerative_cosine = silhouette_score(M, agglomerative_labels, metric='cosine')
                        scores['silhouette_score_agglomerative_cosine'] = silhouette_score_agglomerative_cosine
                        silhouette_score_agglomerative_euclidean = silhouette_score(M, agglomerative_labels, metric='euclidean')
                        scores['silhouette_score_agglomerative_euclidean'] = silhouette_score_agglomerative_euclidean
                    except (ValueError):
                        print("Agglomerative:: Oops! Only one cluster found!") 
                    except:
                        print("Found numpy.linalg.LinAlgError: Singular matrix")

                    dic.append(scores)
                    iter += 1

        mem_dic = {
            "x_train": "x_train",
            "y_pred": "y_pred",
        }
                                            
        # write dictionary to file
        write_dic_to_file(dic, mem_dic, sort_key, directory, filename, baseline, ext)

    end_time = time.time()
    print("Took", (end_time - start_time)/60, "minutes to complete all baseline configs for", filename, "dataset.")
