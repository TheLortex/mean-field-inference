
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
import setproctitle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.set_printoptions(precision=3,suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--boardSz', type=int, default=2)
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--lognmodes', type=int, default=2)
parser.add_argument('--explain', default=False, action='store_true')
parser.add_argument('--n', type=int, default=-1)
args = parser.parse_args()


def judge(args):
    setproctitle.setproctitle('testing {} on {}'.format(args.input, args.dataset))

    with open(args.dataset,'rb') as f:
        dataset_X, dataset_Y = pickle.load(f)

# todo: be backward compatible. (old format = weights, unary, annealing, new format = dictionnary)

    with open(args.input,'rb') as f:
        model_data = pickle.load(f)
    print(model_data['args'])
    print("Version:",model_data['version'])

    zeropad = not(model_data['args'].nopad)

    g = args.boardSz
    n = (g**2)
    if zeropad:
        n *= 2
    p = 1+g**2

    n_samples = len(dataset_X) if args.n == -1 else args.n
    batch_size = args.bs
    n_modes = args.lognmodes

    if zeropad:
        inputs = sudoku.grid_to_clip(sudoku.expand_matrix(dataset_X,g,p))
    else:
        inputs = sudoku.grid_to_clip(dataset_X)
    w = model_data['links']
    u = model_data['unary']
    a = model_data['annealing']

    FNN = model_data['FNN']
    _,_,L2,_ = FNN
    h,k = L2.shape

    links = tf.Variable(tf.convert_to_tensor(w))
    links_sym = links+tf.transpose(links, [0, 1, 2, 4, 3])
    unary = tf.Variable(tf.convert_to_tensor(u))

    mmmf = mf.BatchedMultiModalMeanField(n, n, p, batch_size, links_sym, unary, np.exp(a), a.shape[0], k=k, h=h, FNN=FNN)
    q_mf = mmmf.get_q_mf_values()


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    n_correct = 0
    n_correct_mul = 0
    n_correct_one = 0
    n_correct_tot = 0
    n_mul = 0
    n_one = 0
    n_tot = 0

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
        for modes_clip, q_modes, grid_input, grid_output in zip(mmmf._modes, q_values, dataset_X[b*batch_size:(b+1)*batch_size], dataset_Y[b*batch_size:(b+1)*batch_size]):
            ok = False
            res = []
            if args.explain:
                print("###")
                print(sudoku.infer_grid(grid_input))
                print(sudoku.infer_grid(grid_output))
            for q_mode, clip_grid in zip(q_modes, sudoku.clip_to_grid(modes_clip)):
                grid = sudoku.infer_grid(sudoku.reduce_matrix(q_mode,g,p) if zeropad else q_mode)
                res.append(q_mode)
                if args.explain:
                    print("##")
                    print(sudoku.infer_grid(sudoku.reduce_matrix(clip_grid,g,p) if zeropad else clip_grid))
                    print(np.max(q_mode,axis=-1))
                    print(grid)
                    print("##")
                if sudoku.is_correct(grid,g):
                    ok = True
                    n_correct_tot += 1

            n_sol = sudoku.n_solutions_grid(sudoku.infer_grid(grid_input),g)
            is_one = False#sudoku.n_solutions_grid(sudoku.infer_grid(grid_input),g) == 1
            if is_one:
                n_one += 1
            else:
                n_mul += 1

            n_tot += n_sol

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

        print('{}/{} | S: {}/{} | M: {}/{} | T: {}/{}                  '.format(n_correct,batch_size*(b+1),n_correct_one,n_one,n_correct_mul,n_mul,n_correct_tot,n_tot),end='\r')
        sys.stdout.flush()
    print()
    return (n_correct, n_samples, n_correct_tot, n_tot)


if __name__ == '__main__':
    judge(args)
