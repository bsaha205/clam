import argparse
import os
import numpy as np
from scipy import linalg as la
# from constants import Constants
import time
import scipy.io
from sklearn.datasets import fetch_olivetti_faces

# for classmap
clmap = {}
clcnt = 0

######PARSER########
def parse_args():
    parser = argparse.ArgumentParser(description='am clustering')
    parser.add_argument('-i', dest='fname', help='input file name')
    parser.add_argument('-k', type=int, help='number of clusters')
    parser.add_argument('-scale',
                        dest='scaleVal',
                        default=1,
                        type=float,
                        help='scale value')
    parser.add_argument('-prcomp',
                        default=0,
                        type=float,
                        help='''do PCA: 0 means no PCA, 
                        value in (0,1) means use var thresh,
                        value > 1 should be int and is num components''')
    parser.add_argument('-normalize',
                        default=True,
                        action='store_false',
                        help='turn off normalize points')
    parser.add_argument('-nA',
                        type=int,
                        default=-1,
                        help='number of attractors')
    parser.add_argument('-delimiter',
                        default=',',
                        help='field separator delimiter')
    parser.add_argument('-plot',
                        default=False,
                        action='store_true',
                        help='plot the clusters on 2D plane')
    args = parser.parse_args()

    if not args.fname and not args.k:
        parser.print_usage()
        parser.exit()
    if args.randSeed:
        np.random.seed(args.randSeed)
    if args.nA <= 0:
        args.nA = 10 * args.k  # set to 10 times num clusters

    return args

# for classmap
clmap = {}
clcnt = 0

def class2int(c):
    global clcnt
    c = c.strip('"')
    if c not in clmap:
        clmap[c] = clcnt
        clcnt += 1
    return clmap[c]

def read_file(fname, delimiter, label_filename=None):
    fn, fext = os.path.splitext(fname)

    if fname.startswith("cifar"):
        x_file = open('data/' + fname, 'r')
        label_file = open('data/' + label_filename, 'r')

        X = []
        for x_line, label_line in zip(x_file, label_file):
            a = x_line.strip().split(delimiter)
            cval = class2int(label_line.strip())
            a.append(cval)
            X.append(list(map(float, [float(x) for x in a])))
        
        return np.array(X)

    # elif fext == ".npz":
    #     data = np.load('data/' + fname, allow_pickle=True)
    #     X = data['X']
    #     Y = data['Y']
    #     print("X.shape:", X.shape, "Y.shape:", Y.shape)
    #     return X, Y
    
    elif fext == ".mat":
        f = scipy.io.loadmat('data/' + fname)
        X = f['X']
        Y = f['Y']
        return X, Y

    f = open('data/' + fname, "r")
    X = []
    i = 1
    for l in f.readlines():
        # print('l:', l)
        a = l.strip().split(delimiter)
        if len(a) == 1:
            a = l.strip().split()
        if fname == 'usps.t':
            temp = a[0]
            a[0] =  a[len(a) - 1]
            a[len(a) - 1] = temp
            for i in range(len(a)-1):
                a[i] = a[i].split(':')[1]
        cla = len(a) - 1
        cval = class2int(a[cla])
        a[cla] = cval
        X.append(list(map(float, [float(x) for x in a])))
        
    return np.array(X)

def get_class_count():
    return clcnt

def get_class_map():
    return clmap