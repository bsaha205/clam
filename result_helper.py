from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import time

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering

from hyper_params import *


def get_init_mem(X, k):
    """Performs k-means++/random initialization for k-means clustering.
    Parameters:
    X (numpy array): Data matrix, where each row represents a data point.
    k (int): Number of clusters.
    Returns:
    numpy array: Initial centroids for k-means clustering.
    """
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features))

    if init == 'kmeans++':
        # Choose the first centroid randomly
        centroids[0] = X[np.random.choice(n_samples)]
        for i in range(1, k):
            # Compute the distance between each data point and the nearest centroid
            distances = euclidean_distances(X, centroids[:i]).min(axis=1)
            # Choose the next centroid randomly, weighted by distance from existing centroids
            weights = distances ** 2
            probabilities = weights / np.sum(weights)
            centroids[i] = X[np.random.choice(n_samples, p=probabilities)]
    else:
        centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    return np.reshape(centroids, (k, n_features, 1))

def calculate_dist_to_closest_memory(cluster_center, X):
    no_of_cluster = len(cluster_center)
    n = len(X)
    total_distance = 0

    for i in range(n):
        distance_to_closest_center = 100000
        for j in range(no_of_cluster):
            dist = distance.euclidean(X[i], cluster_center[j])
            if dist < distance_to_closest_center:
                distance_to_closest_center = dist
        total_distance += distance_to_closest_center

    return total_distance

def find_closest_center(cluster_centers, point):
    distance_to_closest_center = 100000
    for j in range(len(cluster_centers)):
        dist = distance.euclidean(point, cluster_centers[j])
        if dist > 0 and dist < distance_to_closest_center:
            closest_cluster = j
            distance_to_closest_center = dist
    return closest_cluster

def adjust_clusters(cluster_center, label_pred):
    no_of_cluster = len(cluster_center)
    print('label_count_total:', len(label_pred))
    print('Original cluster assignment:')
    print_distribution(label_pred, no_of_cluster)
    for i in range(no_of_cluster):
         label_count =  label_pred.count(i)
        #  print('labels.count(', i, '):', label_count)
         if label_count == 0:
            closest_center = find_closest_center(cluster_center, cluster_center[i])
            print('closest_center:', closest_center)
            if label_pred.count(closest_center) > 1:
                first_index = label_pred.index(closest_center)
                print('first_index:', first_index, 'label_pred[first_index]:', label_pred[first_index])
                label_pred[first_index] = i  # set the first occurence of closest center to itself

    print('Distribution after cluster adjusting:')
    print_distribution(label_pred, no_of_cluster)
    return label_pred

def get_final_clusters(cluster_center, X):

    # Calculate the distances between each data point and each cluster center
    distances = distance.cdist(X, cluster_center)

    # Get the index of the closest center for each data point
    label_pred = np.argmin(distances, axis=1).tolist()

    return label_pred


def print_distribution(labels, no_of_cluster):
    for i in range(no_of_cluster):
         print('labels.count(', i, '):', labels.count(i))
        

def print_results(Y, label_pred):
    nmi_score = normalized_mutual_info_score(Y, label_pred)
    print("nmi_score:", nmi_score)

    ari_score = adjusted_rand_score(Y, label_pred)
    print("ari_score:", ari_score)

    ri_score = rand_score(Y, label_pred)
    print("ri_score:", ri_score)


def print_status_bar(iteration, total, loss, metrics=None):
    metrics = ' - '.join(["{}: {:.4f}".format('loss', m) for m in [loss] + (metrics or [])])
    end = '' if iteration < total else '\n'
    print('\r{}/{} - '.format(iteration, total) + metrics, end=end)


def get_basline_cluster_labels(data, no_of_cluster):
   
    # kmeans clustering
    print('Kmeans begins')
    s_time = time.time()
    kmeans = KMeans(n_clusters=no_of_cluster, init='k-means++', random_state=kmeans_random_state_list[0], n_init=kmeans_n_init_list[0]).fit(data)
    kmeans_labels = kmeans.labels_.tolist()
    e_time = time.time()
    print("Took", (e_time - s_time), "seconds to complete kmeans with", kmeans_n_init_list[0], "init")

    # Spectral clustering
    print('spectral begins')
    s_time = time.time()
    spectral = SpectralClustering(n_clusters=no_of_cluster, affinity='nearest_neighbors', gamma=0.05, n_neighbors=10, assign_labels='kmeans', n_init=10, random_state=0).fit(data)
    spectral_labels = spectral.labels_.tolist()
    e_time = time.time()
    print("Took", (e_time - s_time), "seconds to complete spectral with 20 init")

    # Agglomerative clustering
    print('agglomerative begins')
    s_time = time.time()
    agglomerative = AgglomerativeClustering(n_clusters=no_of_cluster, affinity='euclidean', linkage='single').fit(data)
    agglomerative_labels = agglomerative.labels_.tolist()
    e_time = time.time()
    print("Took", (e_time - s_time), "seconds to complete agglomerative")

    return kmeans_labels, spectral_labels, agglomerative_labels


def print_distributions_of_clusters(Y_true, label_pred, kmeans_labels, spectral_labels, agglomerative_labels, no_of_cluster, printed_flag):
   
    print('----------------Print label_pred Info---------------- ')
    print_distribution(label_pred, no_of_cluster)

    if not printed_flag:
        print('----------------Print Y_true Info---------------- ')
        print_distribution(Y_true.tolist(), no_of_cluster)

        print('----------------Print kmeans_labels Info---------------- ')
        print_distribution(kmeans_labels, no_of_cluster)

        print('----------------Print spectral_labels Info---------------- ')
        print_distribution(spectral_labels, no_of_cluster)

        print('----------------Print agglomerative_labels Info---------------- ')
        print_distribution(agglomerative_labels, no_of_cluster)


def print_cluster_similarities_matrices(Y_true, label_pred, kmeans_labels, spectral_labels, agglomerative_labels, printed_flag):
    print('----------AM vs Ytrue----------')
    print_results(Y_true, label_pred)

    if not printed_flag:
        print('----------Kmeans vs Ytrue----------')
        print_results(Y_true, kmeans_labels)

        print('----------Spectral vs Ytrue----------')
        print_results(Y_true, spectral_labels)

        print('----------Agglomerative vs Ytrue----------')
        print_results(Y_true, agglomerative_labels)

    print('----------AM vs Kmeans----------')
    print_results(label_pred, kmeans_labels)

    print('----------AM vs Spectral----------')
    print_results(label_pred, spectral_labels)

    print('----------AM vs Agglomerative----------')
    print_results(label_pred, agglomerative_labels)


def print_cluster_quality_matrices(scores, data, kmeans_labels, spectral_labels, agglomerative_labels):
    try:
        silhouette_score_kemans = silhouette_score(data, kmeans_labels, metric='euclidean')
        print('Silhouette_score of Kmeans:', silhouette_score_kemans)
        scores['silhouette_score_kemans'] = silhouette_score_kemans
    except (ValueError):
        print("Kmeans:: Oops! Only one cluster found!")
    
    try:
        silhouette_score_spectral = silhouette_score(data, spectral_labels, metric='euclidean')
        print('Silhouette_score of Spectral:', silhouette_score_spectral)
        scores['silhouette_score_spectral'] = silhouette_score_spectral
    except (ValueError):
        print("Spectral:: Oops! Only one cluster found!")
    
    try:
        silhouette_score_agglomerative = silhouette_score(data, agglomerative_labels, metric='euclidean')
        print('Silhouette_score of Agglomerative:', silhouette_score_agglomerative)
        scores['silhouette_score_agglomerative'] = silhouette_score_agglomerative
    except (ValueError, RuntimeError, TypeError, NameError):
        print("Agglomerative:: Oops! Only one cluster found!")
