# Hyper Params
mu = 0.1
sigma = 0.1
init = 'gaussian' # 'random', 'kmeans++', 'gaussian'

# clip values
clip_low = 0.001
clip_high = 1000

# model param
beta_list = [1.1, 1.5, 1.9, 2.4] # temperature parameter, varies upon datasets
alpha_list = [.1] # number of steps, varies upon datasets
N_steps_add_list = [0]

batch_size_list = [8] # varies upon datasets
learning_rate_kernel_list = [.001, .01, .1] # varies upon datasets
learning_rate_threshold = 0.000001

# masking param
mask_prob_list = [.2] # # varies upon datasets
mask_value_list = ['mean', 'max', 'min'] # ['mean', 'max', 'min'] 
dim_change_cycle_list = [3]

store_memory = False
epoch_diff = 10
example_size = 10

N_ep = 200 # number of epochs
N_init = 1 # number of restart for the same configuration
total_config = 36
mask = "mask_" + init

max_patience = 5
loss_threshold = 0.001
reduce_frac = 0.8

# baseline list
baseline_list =  ['kmeans', 'spectral', 'agglomerative', 'skmeans']

# kmeans param
kmeans_random_state_list = [0]
kmeans_n_init_list = [1000] 
dataset_list = ['original', 'noisy']
std_list = [.1, .2, .3, .4, .5]

# skmeans param
skmeans_maxiter = 1000
skmeans_tolerance = 0.0001 
skmeans_update_interval_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200]
init_list = ['kmeans', 'random']

# spectral param
spectral_n_init_list = [1000]
spectral_gamma_list = [.001, .01, .05, .1, .5, .75, 1, 2, 5]
spectral_affinity_list = ['nearest_neighbors' , 'rbf', 'precomputed', 'precomputed_nearest_neighbors']
spectral_n_neighbors_list =  [10, 15, 20, 50] # [10, 15, 20, 50]
spectral_assign_labels_list = ['kmeans', "discretize"] # 'kmeans',  "discretize", does not work for 'cluster_qr'

# Agglomerative param
agglomerative_affinity_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
agglomerative_linkage_list = ['single',  'average', 'complete', 'ward']

# DBscan param
dbscan_eps = 3
dbscan_min_samples = 5

# data parameters
filename = 'zoo.csv'
# filename = 'movement_libras.csv'
# filename = 'ecoli.data'
# filename = 'Yale.mat'
# filename = 'mp_exp.txt'
# filename = 'usps.t'
# filename = 'ctg.txt'
# filename = 'segment.dat'
# filename = 'GCM.csv'
# filename = 'fmnist.csv'
# filename = 'BNG_JapaneseVowels_1M_14_9_zipped.npz'
# filename = 'cifar10.pretrained.densenet.imgnet.50000x1920.csv'
label_filename = 'cifar10.pretrained.densenet.imgnet.50000.labels'


directory = "results/original/"
# directory = "results/weighted/"
# directory = "results/baselines"

model_directory = 'saved_models/original_'
# model_directory = 'saved_models/weighted_'
# model_directory = 'saved_models/no_mask_'

suffix = str(N_ep) + "_" + str(total_config) + "_" + str(N_init) + "_" + mask
ext = ".json"
delimiter = ','