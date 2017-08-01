
from __future__ import print_function
import numpy as np
import tensorflow as tf
import mf
import pickle
import sys
import sudoku
import argparse
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.set_printoptions(precision=3,suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--boardSz', type=int, default=2)
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--lognmodes', type=int, default=2)
parser.add_argument('--old', default=False, action='store_true')
parser.add_argument('--out', type=str, default='out')
args = parser.parse_args()

with open(args.dataset,'rb') as f:
    dataset_X, dataset_Y = pickle.load(f)

with open(args.input,'rb') as f:
    w, u, a = pickle.load(f)

g = args.boardSz
n = 2*(g**2)
p = 1+g**2

n_samples = len(dataset_X)
batch_size = args.bs
n_modes = args.lognmodes

inputs = sudoku.grid_to_clip(sudoku.expand_matrix(dataset_X,g,p))

links = tf.Variable(tf.convert_to_tensor(w))
links_sym = links+tf.transpose(links, [0, 1, 3, 2])
unary = tf.Variable(tf.convert_to_tensor(u))

mmmf = mf.BatchedMultiModalMeanField(n, n, p, batch_size, links_sym, unary, np.exp(a), a.shape[0])
q_mf = mmmf.get_q_mf_values()
E_mf = mmmf.get_energy_values()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

n_correct = 0
n_correct_mul = 0
n_correct_one = 0
n_mul = 0
n_one = 0

correct_energy_table    = []
incorrect_energy_table  = []


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.transpose(np.transpose(e_x,(2,0,1)) / e_x.sum(axis=-1),(1,2,0))

def compute_circular_convolution(image, filtr):
    image_1 = tf.concat(2*[image], axis=1)
    image_2 = tf.concat(2*[image_1], axis=2)
    image_3 = tf.slice(image_2, [0,0,0,0], [-1, 2*n-1, 2*n-1, -1])
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    return tf.nn.conv2d(image_3, filtr, strides, padding)

image = tf.placeholder(tf.float32, [1,n,n,p])
filtr = tf.placeholder(tf.float32, [n,n,p,p])
res = compute_circular_convolution(image, filtr)

def circular_convolution(np_image, np_filtr):
    global image, filtr, sess, res
    return sess.run(res, feed_dict={image: np.expand_dims(np_image, axis=0), filtr: np_filtr})[0]

def compute_energy(q,links,unary):
    E = circular_convolution(q, links)
    return np.sum((E+unary)*q)

for b in range(n_samples/batch_size):
    mmmf.reset_all(np.expand_dims(np.array(inputs[b*batch_size:(b+1)*batch_size]),axis=1))
    for _ in range(n_modes):
        mmmf.iteration(sess)

    parameters = {
        mmmf._theta_clip: np.reshape(mmmf._modes,(-1,2,n,n,p)),
        mmmf._T: 0.2
    }
    q_values = sess.run(q_mf, feed_dict=parameters)
    q_values = np.reshape(q_values,(batch_size,-1,n,n,p))
    for q_modes, grid_input in zip(q_values, dataset_X[b*batch_size:(b+1)*batch_size]):
        for q_mode in q_modes:
            
            grid = sudoku.infer_grid(sudoku.reduce_matrix(q_mode,g,p))
            if args.old:
                inp = q_mode
            else:
                inp = sudoku.to_prob(grid,p)
            E_mode = compute_energy(sudoku.expand_matrix(np.array([inp]),g,p)[0],w,u)
            if sudoku.is_correct(grid,g):
                correct_energy_table.append(E_mode)
            else:
                incorrect_energy_table.append(E_mode)

    max_correct = max(correct_energy_table)
    min_incorrect = min(incorrect_energy_table)
    lvl = 50
    mid_energy = (np.percentile(correct_energy_table,100-lvl) + np.percentile(incorrect_energy_table,lvl))/2.
    print(max_correct,mid_energy,min_incorrect,sum(correct_energy_table < mid_energy),'/',len(correct_energy_table),sum(incorrect_energy_table > mid_energy),'/',len(incorrect_energy_table),end='\r')
    sys.stdout.flush()
with open(args.out,'wb') as f:
    pickle.dump((correct_energy_table,incorrect_energy_table),f)

print()
with open('_success_one.pkl','wb') as f:
    pickle.dump(success_one, f)
with open('_success_mul.pkl','wb') as f:
    pickle.dump(success_mul, f)
with open('_failure_one.pkl','wb') as f:
    pickle.dump(failure_one, f)
with open('_failure_mul.pkl','wb') as f:
    pickle.dump(failure_mul, f)

