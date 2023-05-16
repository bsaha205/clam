import numpy as np;  
import matplotlib.pyplot as plt 
from scipy.spatial import Voronoi, voronoi_plot_2d
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from scipy.spatial import distance
from matplotlib.pyplot import figure
import time
from scipy.stats import multivariate_normal
import random
import math
import os
from sklearn.cluster import KMeans
from s_kmeans import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from collections import Counter

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
sc = silhouette_score

def get_labels(M, memories):
    labels = []
    for i in range(M.shape[0]):
        min_dist = 1000
        for j in range(len(memories)):
            euclid_dist = distance.euclidean(M[i], memories[j])
            if (euclid_dist < min_dist):
                min_dist = euclid_dist
                label = j
        labels.append(label)
    
    return labels

def generate_data(xpoints, ypoints):
    M = []
    xstep = 1/xpoints
    ystep = 1/ypoints
    
    x = 0
    for i in range(xpoints):
        grid_x = []
        grid_y = []
        y = 0
        for j in range(ypoints):
            grid_x.append(x)
            grid_y.append(y)
            y += ystep

        M.append(np.stack((grid_x, grid_y), axis=1))
        x += xstep
    
    M = np.array(M)
    M = np.reshape(M, (xpoints*ypoints, 2)) 
    return M

def plot_graph(labels, true_memories, learned_memories, algo, sc_score, d_type):

    if d_type == 'original':
        data = M
        d_dir = data_dir
        f_dir = fig_dir
    else:
        data = M_noisy
        d_dir = noisy_data_dir
        f_dir = noisy_fig_dir
    # Writing to data.json
    dic = {}
    dic['data'] = data.tolist()
    dic['labels'] = labels
    dic['true_memories'] = memories.tolist()
    if type(learned_memories) is list:
        dic['learned_memories'] = learned_memories
    else:
         dic['learned_memories'] = learned_memories.numpy().tolist()
    json_object = json.dumps(dic, indent=4)

    with open(d_dir + algo + '.json', "w") as outfile:
        outfile.write(json_object)


    # plotting
    n_memories = len(true_memories)
    for i in range(data.shape[0]):
        plt.scatter(data[i][0], data[i][1], edgecolor=color_list[labels[i]], facecolor=color_list[labels[i]], s=point_size)
        
    for ii in range(n_memories):
        plt.scatter(learned_memories[ii][0], learned_memories[ii][1], c=color_list[n_memories], s=mem_size, marker='x')
        plt.scatter(true_memories[ii][0], true_memories[ii][1], c=color_list[n_memories], s=mem_size, marker='o')

    plt.xticks([0 , 0.5, 1])
    plt.yticks([0 , 0.5, 1])
    plt.savefig(f_dir + algo + '_' + str(round(sc_score, 3)) + '.png')
    plt.show()


mu = 0.2
sigma = 0.1

class MHN_WITH_1_HIDDEN_LAYER(tf.keras.layers.Layer):
    def __init__(self, N1, N2, beta, alpha, memories, c=1, **kwargs):
        super().__init__(**kwargs)
        self.N1 = N1
        self.N2 = N2
        self.c = c
        self.beta = beta
        self.alpha = alpha
        self.memories = memories

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.N2, self.N1, 1], 
            initializer=RandomNormal(mean=mu, stddev=sigma, seed=None) 
        )
        super().build(input_shape)

    def call(self, v, mask):
        
        Mem = self.kernel
        v = tf.expand_dims(v, axis=-1)
        v = tf.transpose(v, perm=[2, 1, 0])
        diff = Mem - v 
        
        # original clam
        exp_sum_diff = tf.exp(-self.beta/2*tf.reduce_sum(diff**2, axis=1))
        den = tf.expand_dims(tf.reduce_sum(exp_sum_diff, axis=0),axis=0)
        num = tf.reduce_sum(diff*tf.expand_dims(exp_sum_diff,axis=1),axis=0) 
        update = num/den

        # with mask
        mask = tf.transpose(tf.expand_dims(mask, axis=0), perm=[0, 2, 1])
        v += self.alpha*tf.expand_dims(update, axis=0) * mask
        v = tf.transpose(v, perm=[2, 1, 0])
        v = tf.squeeze(v)
        return v

def mean_squared_error(y_true, y_pred, mask):
    loss = tf.reduce_mean(tf.square(y_true-y_pred)*mask)
    return loss

def train_am(model, learning_rate_kernel):
    learning_rate_threshold = 0.000001
    optimizer = Adam(learning_rate=learning_rate_kernel)
    prev_mean_loss = 100000
    patience = 0
    optimum_lr = learning_rate_kernel
    optimum_epoch = 1
    losses_schedular = []
    losses = []
    epochs = []
    min_loss = 10000
    sum_losses = 0
    initial_loss = 0
    nan_flag = False
    
    N_ep = 10
    max_patience = 5
    loss_threshold = 0.001
    reduce_frac = 0.8
    clip_low = 0.001
    clip_high = 1000

    for epoch in range(1, N_ep+1):
        # print(f'Epoch {epoch}/{N_ep}')
        sum_loss = 0
        isFirst =  True
        for step, (x_train, y_train) in enumerate(train_dataset):

            #for tabular data
            x_train = tf.cast(x_train,dtype=tf.float32)
            y_train = tf.cast(y_train,dtype=tf.float32)
               
            # with mask training
            x_train_masked = x_train + np.random.normal(0, 0.01, size=(x_train.shape[0], 2))
            mask = tf.cast(tf.equal(x_train, y_train), dtype=tf.float32)

            # update weights
            with tf.GradientTape() as tape:
                y_pred = model([x_train_masked, mask])
                loss = mean_squared_error(y_train, y_pred, mask)

            sum_loss += loss

            gradients = tape.gradient(loss, model.trainable_variables)

            optimizer.learning_rate.assign(learning_rate_kernel)  # for optimizing memories
            optimizer.apply_gradients(zip(gradients[:1], model.trainable_variables[:1]))
   
            for _ in range(1, len(model.trainable_variables)):
                model.trainable_variables[_].assign(tf.clip_by_value(model.trainable_variables[_], clip_low, clip_high))

        sum_loss = tf.sqrt(sum_loss)
        metrics = ' - '.join(["{}: {:.4f}".format('sum_loss', m) for m in [sum_loss]])
        # print(metrics)

        if math.isnan(sum_loss):
            nan_flag = True
            print('----------- Got nan value for beta', beta, ': Aborting the config.. ----------- ')
            return False

        sum_losses += sum_loss
        losses_schedular.append(sum_loss)

        losses.append(sum_loss)
        epochs.append(epoch)

        if epoch == 1:
            initial_loss = sum_loss

        if sum_loss < min_loss:
            min_loss = sum_loss
            model.save(model_directory)
            optimum_lr = learning_rate_kernel
            optimum_epoch = epoch

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

    return True
            
def evolution(model, memories, d_type):
    labels = []
    for step, (x_train, y_train) in enumerate(test_dataset):
        x_train = tf.cast(x_train,dtype=tf.float32)
        y_train = tf.cast(y_train,dtype=tf.float32)

        mask = tf.cast(tf.equal(x_train, y_train), dtype=tf.float32)

        y_pred = model([x_train, mask])
        
        KS = model.layers[2].get_weights()[0]
        KS = tf.squeeze(KS)
        labels += get_labels(y_pred, KS)

    num_of_cluster_found = len(Counter(labels).keys())
    if(num_of_cluster_found == 1): 
        print('----------- Found only one cluster! -----------')
        return False
    
    if d_type == 'original':
        data = M
    else:
        data = M_noisy
    
    sc_score = sc(data, labels)
    print('----------- sc_am = %.4f' % sc_score, 'alpha:', alpha, 'learning_rate:', learning_rate_kernel, 'beta:', beta)
    sc_list.append(sc_score)
    algo = 'AM_' + d_type + str(alpha) + '_' + str(learning_rate_kernel) + '_' + str(beta)
    plot_graph(labels, memories, KS, algo, sc_score, d_type)
    return True


def am_evolution(xx, yy, beta, alpha, index, learning_rate_kernel, d_type):
    N1 = 2
    N2 = len(xx)
    input_shape = N1
    N_steps = int(1/alpha)
    
    memories = np.stack((xx,yy), axis=1); #combine x and y coordinates
    memories = tf.cast(memories, dtype='float32')
    
    # define model
    input_mask = Input(shape=[input_shape])
    input1 = Input(shape=[input_shape])
    MHN_cell = MHN_WITH_1_HIDDEN_LAYER(N1, N2, beta, alpha, memories)
    x = MHN_cell(input1, input_mask)
   
    for i in range(N_steps-1):
        x = MHN_cell(x, input_mask)

    model = Model(inputs=[input1, input_mask], outputs=x)
    
    # training
    if(not train_am(model, learning_rate_kernel)): return False
    
    # inference
    if not os.path.exists(model_directory):
        return False
    
    model = tf.keras.models.load_model(model_directory, compile=False)
    return evolution(model, memories, d_type)


def read_file(fname, delimiter):
    f = open('data/' + fname, "r")
    X = []
    for l in f.readlines():
        a = l.strip().split(delimiter)
        X.append(list(map(float, [float(x) for x in a])))

    return np.array(X)


def voronoi_tessellation(xx, yy, M):
    numbPoints = len(xx)
    xxyy = np.stack((xx,yy), axis=1); #combine x and y coordinates

    ##Perform Voroin tesseslation using built-in function
    voronoiData=Voronoi(xxyy)

    #create voronoi diagram on the point pattern
    voronoi_plot_2d(voronoiData, ax=ax, show_points=False, show_vertices=False, point_size=1); 

    plt.xlim(-.5, 1.5)
    plt.ylim(-.5, 1.5)


def calculate_scores(data_dir, files):
    data = json.load(open(data_dir + 'original.json'))
    M_ = data['data']
    true_labels = data['labels']
    for i in range(len(files)):
        data = json.load(open(data_dir + files[i] + '.json'))
        labels = data['labels']
        sc_score = sc(M_, labels)
        nmi_score = nmi(true_labels, labels)
        ari_score = ari(true_labels, labels)
        print(files[i], ':: sc:', sc_score, 'nmi:', nmi_score, 'ari:', ari_score)


def plot_graphs(data_dir, fig_dir, algos):
    true_labels = []
    for i in range(len(algos)):
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        data = json.load(open(data_dir + algos[i] + '.json'))
        M_ = data['data']
        labels_ = data['labels']
        
        if algos[i] == 'original':
            true_memories_ = data['memories']
            true_labels = labels_
        else:
            true_memories_ = data['true_memories']
            learned_memories_ = data['learned_memories']
        
        # # for noisy data, we need to remove the outlier
        # del M_[first_ind_outlier:]
        
        colors = []
        for j in range(len(labels_)):
            colors.append(color_list[labels_[j]])
        M_tanspose = np.transpose(M_)
        plt.scatter(M_tanspose[0], M_tanspose[1], color=colors, s=point_size)

        for ii in range(len(true_memories_)):
            # plt.scatter(true_memories_[ii][0], true_memories_[ii][1], c=color_list[len(true_memories_)], s=mem_size, marker='o')
            if algos[i] != 'original':
                plt.scatter(learned_memories_[ii][0], learned_memories_[ii][1], c=color_list[len(true_memories_)], s=mem_size, marker='o', alpha=.8)
                nmi_score = nmi(true_labels, labels_)
                ari_score = ari(true_labels, labels_)
                sc_score = sc(M_, labels_, metric='mahalanobis')
                # sc_score = sc(M_, labels_)
                text = 'NMI: ' + str(round(nmi_score, 3))
                plt.text(.3, .9, text, fontsize = 14)
                text = 'ARI: ' + str(round(ari_score, 3))
                plt.text(.3, .84, text, fontsize = 14)
                text = 'SC: ' + str(round(sc_score, 3))
                plt.text(.3, .78, text, fontsize = 14)

    
        plt.xlim([0, 1]) #original
        plt.ylim([0, 1])
        plt.title('ClAM_clean', fontsize=14)
        plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        plt.savefig(fig_dir + algos[i] + '.png', bbox_inches='tight')




## main function starts here
start_time = time.time()
diff = 0.005
random_seed=1000
color_list = ['green', 'orange', 'm', 'black', '#f2ab15', 'black', 'm', 'orange', 'brown', 'black', 'red']
xx_list = [[.2, .5, .8]]
yy_list = [[.5, .5, .5]]
# xx_list = [[.3, .45, .65]]
# yy_list = [[.6, .15, .6]]

memories = np.stack((xx_list[0],yy_list[0]), axis=1); #combine x and y coordinates
print('len(memories):', len(memories))
print('memories:', memories)
    
point_size = 10 #30
mem_size = 70 #210
point_range_start_list = [250, 100, 200]
point_range_end_list = [300, 150, 250]
# cov_val = [.01, .01] # working covariances
# fixed_cov_val = 0.0105 # working fixed_cov_val
cov_val = [.01, .01, .01] # working
fixed_cov_val = 0.0105 # working
 
fig, ax = plt.subplots(1, 1, figsize=(5,5))

labels = []
for idx, val in enumerate(cov_val):
    cov = np.array([[fixed_cov_val, val], [val, fixed_cov_val]])
    # Generating a Gaussian bivariate distribution with given mean and covariance matrix
    distr = multivariate_normal(cov = cov, mean = memories[idx], allow_singular=True, seed = random_seed)
    # Generating samples out of the distribution
    data = distr.rvs(size = random.randrange(point_range_start_list[idx], point_range_end_list[idx]))
    if idx == 0:
        M = data
    else:
        M = np.concatenate((M, data), axis = 0)
    labels.append([idx]*len(data))
    
    plt.scatter(data[:,0], data[:,1], c=color_list[idx], s=point_size)

for ii in range(len(memories)):
    plt.scatter(memories[ii][0], memories[ii][1], c=color_list[len(memories)], s=mem_size)

# create noisy data
outliers = []
# num_of_outliers = int(M.shape[0]*.05)
num_of_outliers = 3
print('num_of_outliers:', num_of_outliers)

outliers.append((-.05, -.7))
outliers.append((.1, .1))
outliers.append((.3, -.5))

outliers = np.array(outliers)
M_noisy= np.concatenate((M, outliers), axis = 0)
label_noisy = labels.copy()
label_noisy.append([len(cov_val)+1]*len(outliers))

print('M.shape:', M.shape)
print('M_noisy.shape:', M_noisy.shape)
            
# AM parameters
alpha_list = [.1, .05] # [.1, .05]
learning_rate_kernel_list =  [0.001, .01, .1] # [0.001, .01, .1]
beta_list = [.001, .01, .05, .1, .5, 1, 2, 5, 10, 20, 30, 50, 100, 200] # [.001, .01, .05, .1, 1, 5, 10, 30, 50, 100, 200]
batch_size = 32

number_of_points = '(.3k-.5k)_'
suffix = str(len(alpha_list)) + '_' + str(len(learning_rate_kernel_list)) + '_' + str(len(beta_list))
subdir = 'plots/2d_plots/_soft_kmeans/3_clusters/train_' + number_of_points + suffix + '/'
model_base = 'plots/2d_plots/saved_models/train_' + number_of_points + suffix
sc_list = []

data_dir = subdir + 'original/data/'
fig_dir = subdir + 'original/figures/'
noisy_data_dir = subdir + 'noisy/data/'
noisy_fig_dir = subdir + 'noisy/figures/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(noisy_data_dir):
    os.makedirs(noisy_data_dir)
if not os.path.exists(noisy_fig_dir):
    os.makedirs(noisy_fig_dir)


# # read data from json file and plot the graphs
# # for clean data
# data_dir = 'plots/2d_plots/_rebuttal/2_clusters/train_(.3k-.5k)_2_3_11/original/data/'
# fig_dir = 'plots/2d_plots/_rebuttal/2_clusters/train_(.3k-.5k)_2_3_11/original/figures/new/'
# algos = ['original', 'AM_original0.05_0.1_0.01'] # ['AM_original0.05_0.1_0.01', 'Kmeans', , '_AM_0.2_0.1_0.05']
# plot_graphs(data_dir, fig_dir, algos)

# # for noisy data
# data_dir = 'plots/2d_plots/_rebuttal/2_clusters/train_(.3k-.5k)_2_3_11/noisy/data/'
# fig_dir = 'plots/2d_plots/_rebuttal/2_clusters/train_(.3k-.5k)_2_3_11/noisy/figures/new/'
# algos = ['noisy', 'AM_noisy0.05_0.1_0.001'] # ['noisy', 'AM_noisy0.05_0.1_0.001']
# plot_graphs(data_dir, fig_dir, algos)

# # calculate sc, nmi, ari scores from data
# data_dir = 'plots/2d_plots/_soft_kmeans/workable_final_train_(.2k-.4k)_3_3_11/data/'
# files = ['Kmeans', 'Soft-Kmeans (DCEC version)', '_AM_0.2_0.1_0.05']
# calculate_scores(data_dir, files)


# plt original data
plt.xticks([0 , 0.5, 1, 1])
plt.yticks([0 , 0.5, 1, 1])
plt.savefig(fig_dir + 'original.png')

# Writing original data to data.json
labels = [item for sublist in labels for item in sublist]
dic = {}
dic['data'] = M.tolist()
dic['labels'] = labels
dic['memories'] = memories.tolist()
json_object = json.dumps(dic, indent=4)
with open(data_dir + 'original.json', "w") as outfile:
    outfile.write(json_object)


# # plot noisy data
# label_noisy = [item for sublist in label_noisy for item in sublist]
# fig, ax = plt.subplots(1, 1, figsize=(5,5))
# for i in range(M_noisy.shape[0]):
#     plt.scatter(M_noisy[i,0], M_noisy[i,1], c=color_list[label_noisy[i]], s=point_size)

# for ii in range(len(memories)):
#     plt.scatter(memories[ii][0], memories[ii][1], c=color_list[len(memories)], s=mem_size)

# # plt.scatter(outliers[:,0], outliers[:,1], c=color_list[i], s=point_size)
# plt.xticks([0 , 0.5, 1, 1.5])
# plt.yticks([0, 0.5, 1, 1.5])
# plt.savefig(noisy_fig_dir + 'noisy.png')
# # plt.show()

# # Writing noisy data to data.json
# dic = {}
# dic['data'] = M_noisy.tolist()
# dic['labels'] = label_noisy
# dic['memories'] = memories.tolist()
# json_object = json.dumps(dic, indent=4)
# with open(noisy_data_dir + 'noisy.json', "w") as outfile:
#     outfile.write(json_object)



# Kmeans
print('Kmeans original begins')
s_time = time.time()
kmeans_n_init = 1000
kmeans = KMeans(n_clusters=len(memories), init='k-means++', random_state=0, n_init=kmeans_n_init, max_iter=1000).fit(M)
kmeans_labels = kmeans.labels_.tolist()
kmeans_cluster_centers = kmeans.cluster_centers_.tolist()
print('---- sc_kmeans = %.4f' % sc(M, kmeans_labels))
fig, ax = plt.subplots(1, 1, figsize=(5,5))
plot_graph(kmeans_labels, memories, kmeans_cluster_centers, 'Kmeans_original', sc(M, kmeans_labels), 'original')
e_time = time.time()
print("Took", (e_time - s_time), "seconds to complete kmeans with", kmeans_n_init, "init")

# print('Kmeans noisy begins')
# s_time = time.time()
# kmeans_n_init = 1000
# kmeans = KMeans(n_clusters=len(memories), init='k-means++', random_state=0, n_init=kmeans_n_init, max_iter=1000).fit(M_noisy)
# kmeans_labels = kmeans.labels_.tolist()
# kmeans_cluster_centers = kmeans.cluster_centers_.tolist()
# print('---- sc_kmeans = %.4f' % sc(M_noisy, kmeans_labels))
# fig, ax = plt.subplots(1, 1, figsize=(5,5))
# plot_graph(kmeans_labels, memories, kmeans_cluster_centers, 'Kmeans_noisy', sc(M_noisy, kmeans_labels), 'noisy')
# e_time = time.time()
# print("Took", (e_time - s_time), "seconds to complete kmeans with", kmeans_n_init, "init")


# # Soft-Kmeans (DCEC version)
# data_dir = 'plots/2d_plots/_soft_kmeans/3_clusters/train_(.3k-.5k)_2_3_11/data/'
# fig_dir = 'plots/2d_plots/_soft_kmeans/3_clusters/train_(.3k-.5k)_2_3_11/figures/'
# data = json.load(open(data_dir + 'original.json'))
# M = np.array(data['data'])
# print('Soft-Kmeans begins')
# s_time = time.time()
# labels, cluster_centers = DCEC_clustering(M, len(memories), skmeans_update_interval_list[0], init_list[0])
# soft_kmeans_labels = labels.tolist()
# print('---- sc_skmeans = %.4f' % sc(M, soft_kmeans_labels))
# cluster_centers = np.asarray(cluster_centers).reshape([len(memories), 2])
# soft_kmeans_cluster_centers = cluster_centers.tolist()
# fig, ax = plt.subplots(1, 1, figsize=(5,5))
# plot_graph(soft_kmeans_labels, memories, soft_kmeans_cluster_centers, 'Soft-Kmeans (DCEC version)', sc(M, soft_kmeans_labels))
# e_time = time.time()
# print("Took", (e_time - s_time), "seconds to complete Soft-Kmeans DCEC version")

# AM evolution
# for original data
dataset = tf.data.Dataset.from_tensor_slices((M, M))
train_dataset = dataset.shuffle(M.shape[0]).batch(batch_size)
test_dataset = dataset.batch(batch_size)
for i in range(len(xx_list)):
    for alpha in alpha_list:
        for learning_rate_kernel in learning_rate_kernel_list:
            for beta in beta_list:
                print("alpha:", alpha, " beta:", beta, " learning_rate_kernel:",learning_rate_kernel)
                model_directory = model_base + str(alpha) + '_' + str(learning_rate_kernel) + '_' + str(beta)
                fig, ax = plt.subplots(1, 1, figsize=(5,5))

                s_time = time.time()
                # am evolution
                if(not am_evolution(xx_list[i], yy_list[i], beta, alpha, i, learning_rate_kernel, 'original')):
                    continue
                
                e_time = time.time()  

# # for noisy data
# dataset = tf.data.Dataset.from_tensor_slices((M_noisy, M_noisy))
# train_dataset = dataset.shuffle(M.shape[0]).batch(batch_size)
# test_dataset = dataset.batch(batch_size)
# for i in range(len(xx_list)):
#     for alpha in alpha_list:
#         for learning_rate_kernel in learning_rate_kernel_list:
#             for beta in beta_list:
#                 print("alpha:", alpha, " beta:", beta, " learning_rate_kernel:",learning_rate_kernel)
#                 model_directory = model_base + str(alpha) + '_' + str(learning_rate_kernel) + '_' + str(beta)
#                 fig, ax = plt.subplots(1, 1, figsize=(5,5))

#                 s_time = time.time()
#                 # am evolution
#                 if(not am_evolution(xx_list[i], yy_list[i], beta, alpha, i, learning_rate_kernel, 'noisy')):
#                     continue
                
#                 e_time = time.time()  


end_time = time.time()
print("Took", (end_time - start_time)/60, "minutes to complete all", len(learning_rate_kernel_list)*len(alpha_list)*len(beta_list), "config")