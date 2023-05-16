import numpy as np;  
import matplotlib.pyplot as plt 
from scipy.spatial import Voronoi, voronoi_plot_2d
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Model
from scipy.spatial import distance
from matplotlib.pyplot import figure
import time
import os
import json

def get_labels(M, memories):
    labels = []
    for i in range(M.shape[0]):
        min_dist = 100000
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

def voronoi_tessellation(xx, yy, M):
    numbPoints = len(xx)
    xxyy = np.stack((xx,yy), axis=1); #combine x and y coordinates
    print('xxyy:', xxyy)

    ##Perform Voroin tesseslation using built-in function
    voronoiData=Voronoi(xxyy)

    #create voronoi diagram on the point pattern
    voronoi_plot_2d(voronoiData, ax=ax, show_points=False, show_vertices=False, line_width=line_width, point_size=1); 

    plt.xlim(-.01, 1)
    plt.ylim(-.01, 1)


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
        self.kernel = tf.expand_dims(self.memories, axis=-1)
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


def evolution(model, memories, index, beta, alpha):
    
    labels = []
    for step, (x_train, y_train) in enumerate(test_dataset):
        x_train = tf.cast(x_train,dtype=tf.float32)
        y_train = tf.cast(y_train,dtype=tf.float32)

        mask = tf.cast(tf.equal(x_train, y_train), dtype=tf.float32)

        y_pred = model([x_train, mask])
        labels += get_labels(y_pred, memories)

    filename = data_dir + 'voronoi_am_' + str(index) + '_' + str(int(1/alpha)) + '_' + str(beta)
    dic = {}
    dic['data'] = M.tolist()
    dic['labels'] = labels
    dic['memories'] = memories.numpy().tolist()
    json_object = json.dumps(dic, indent=4)

    # Writing to data.json
    with open(filename + '.json', "w") as outfile:
        outfile.write(json_object)

    print("----starting to plot the points-----")
    filename = fig_dir + 'voronoi_am_' + str(index) + '_' + str(int(1/alpha)) + '_' + str(beta)
    colors = []
    for i in range(len(labels)):
        colors.append(color_list[labels[i]])
    M_tanspose = np.transpose(M)
    plt.scatter(M_tanspose[0], M_tanspose[1], color=colors, s=point_size)

    for ii in range(len(memories)):
        plt.scatter(memories[ii][0], memories[ii][1], c=color_list[len(memories)], s=mem_size)
    
    plt.axis('off')
    plt.savefig(filename + '.png', bbox_inches='tight')


def am_evolution(xx, yy, beta, alpha, index):

    # # read data from json file and plot the graphs
    # filename = data_dir + 'voronoi_am_' + str(index) + '_' + str(int(1/alpha)) + '_' + str(beta)
    # data = json.load(open(filename + '.json'))
    # M_ = data['data']
    # labels_ = data['labels']
    # memories_ = data['memories']
  
    # filename = fig_dir + 'voronoi_am_' + str(index) + '_' + str(int(1/alpha)) + '_' + str(beta)
    # colors = []
    # for i in range(len(labels_)):
    #     colors.append(color_list[labels_[i]])
    # M_tanspose = np.transpose(M_)
    # plt.scatter(M_tanspose[0], M_tanspose[1], color=colors, s=point_size)

    # for ii in range(len(memories_)):
    #     plt.scatter(memories_[ii][0], memories_[ii][1], c=color_list[len(memories_)], s=mem_size)
    #     # plt.text(memories_[ii][0]+diff, memories_[ii][1]+diff, str(ii+1), color=color_list[len(memories_)+1], fontsize=12)
    
    # # plt.title('AM Partition: step = ' + str(1/alpha) + ', beta = ' + str(beta))
    # plt.axis('off')
    # plt.savefig(filename + '.png', bbox_inches='tight')

    
    # training code
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
    
    # inference
    evolution(model, memories, index, beta, alpha)



## Start of the main function
start_time = time.time()

# Simulation window parameters
xMin = 0
xMax = 1
yMin = 0
yMax = 1

# rectangle dimensions
xDelta = xMax - xMin; #width
yDelta = yMax - yMin #height
areaTotal = xDelta * yDelta; 

# parameters
diff = 0.005
color_list = ['green', 'blue', 'orange', 'brown', 'm', 'black', 'red']
xx_list = [[.1, .2, .4, .6, .9]]
yy_list = [[.3, .1, .7, .5, .2]]
xpoints = 500
ypoints = 500
point_size = .4
mem_size = 128
line_width = 3
fig_size = 5
data_dir = 'plots/2d_plots/_fixed_memory_voronoi_am/' + str(xpoints) + 'x' + str(ypoints) + '/memory_' + str(len(xx_list[0])) + '/data/'
fig_dir = 'plots/2d_plots/_fixed_memory_voronoi_am/' + str(xpoints) + 'x' + str(ypoints) + '/memory_' + str(len(xx_list[0])) + '/figures/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

M = generate_data(xpoints, ypoints)
print("len(xx_list):", len(xx_list))
print("M.shape:", M.shape)


beta_list = [5, 15, 30, 40, 75] #[.0001, .001, .01, .1, 1, 10, 20, 50, 100, 150, 200]
alpha_list = [.1]
batch_size = 512
dataset = tf.data.Dataset.from_tensor_slices((M, M))
test_dataset = dataset.batch(batch_size)

for i in range(len(xx_list)):
    for alpha in alpha_list:
        for beta in beta_list:
            print("alpha:", alpha, " beta:",beta)
            fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))

            s_time = time.time()
            # voronoi tessellation
            voronoi_tessellation(xx_list[i], yy_list[i], M)
            
            # am evolution
            am_evolution(xx_list[i], yy_list[i], beta, alpha, i)
            
            e_time = time.time()
            print("Took", e_time - s_time, "seconds to complete one config")
            
end_time = time.time()
print("Took", (end_time - start_time)/60, "minutes to complete all", len(xx_list)*len(alpha_list)*len(beta_list), "config")
    
