
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


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

n_correct = 0
n_correct_mul = 0
n_correct_one = 0
n_mul = 0
n_one = 0

success_mul = []
success_one = []
failure_mul = []
failure_one = []

for b in range(n_samples/batch_size):
    mmmf.reset_all(np.expand_dims(np.array(inputs[b*batch_size:(b+1)*batch_size]),axis=1))
    for _ in range(n_modes):
        mmmf.iteration(sess)

    parameters = {
        mmmf._theta_clip: np.reshape(mmmf._modes,(-1,2,n,n,p)),
        mmmf._T: 1.
        }
    q_values = np.reshape(sess.run(q_mf, feed_dict=parameters),(batch_size,-1,n,n,p))
    for q_modes, grid_input in zip(q_values, dataset_X[b*batch_size:(b+1)*batch_size]):
        ok = False
        res = []
        for q_mode in q_modes:
            grid = sudoku.infer_grid(sudoku.reduce_matrix(q_mode,g,p))
            res.append(q_mode)
            if sudoku.is_correct(grid,g):
                ok = True
                break

        is_one = False#sudoku.n_solutions_grid(sudoku.infer_grid(grid_input),g) == 1
        if is_one:
            n_one += 1
        else:
            n_mul += 1

        res = np.array(res)
        if ok:
            if is_one:
                success_one.append((grid_input,res))
                n_correct_one += 1
            else:
                success_mul.append((grid_input,res))
                
                n_correct_mul += 1
            n_correct += 1
        else:
            if is_one:
                failure_one.append((grid_input,res))
            else:
                failure_mul.append((grid_input,res))

    print('{}/{} | S: {}/{} | M: {}/{}                   '.format(n_correct,batch_size*(b+1),n_correct_one,n_one,n_correct_mul,n_mul),end='\r')
    sys.stdout.flush()
print()
with open('_success_one.pkl','wb') as f:
    pickle.dump(success_one, f)
with open('_success_mul.pkl','wb') as f:
    pickle.dump(success_mul, f)
with open('_failure_one.pkl','wb') as f:
    pickle.dump(failure_one, f)
with open('_failure_mul.pkl','wb') as f:
    pickle.dump(failure_mul, f)

