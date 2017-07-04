import numpy as np
import tensorflow as tf
import mf
import pickle
import sys
import sudoku

np.set_printoptions(precision=3,suppress=True)

with open(sys.argv[1],'rb') as f:
    dataset_X, dataset_Y = pickle.load(f)

with open(sys.argv[2],'rb') as f:
    w, u = pickle.load(f)

g = 3
n = 2*(g**2)
p = 1+g**2

n_samples = 10000
batch_size = 50
n_modes = 10

inputs = sudoku.grid_to_clip(sudoku.expand_matrix(dataset_X,g,p))

links = tf.Variable(tf.convert_to_tensor(w))
links_sym = links+tf.transpose(links, [0, 1, 3, 2])
unary = tf.Variable(tf.convert_to_tensor(u))

mmmf = mf.BatchedMultiModalMeanField(n, n, p, batch_size, links_sym, unary)
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
    print(q_values.shape)
    for q_modes, grid_input in zip(q_values, dataset_X[b*batch_size:(b+1)*batch_size]):
        ok = False
        for q_mode in q_modes:
            grid = sudoku.infer_grid(sudoku.reduce_matrix(q_mode,g,p))
            if sudoku.is_correct(grid,g):
                ok = True
                break
        if g == 2:
            is_one = sudoku.n_solutions_grid(sudoku.infer_grid(grid_input)) == 1
        else:
            is_one = False

        if is_one:
            n_one += 1
        else:
            n_mul += 1

        if ok:
            if is_one:
                success_one.append(grid_input)
                n_correct_one += 1
            else:
                success_mul.append(grid_input)
                n_correct_mul += 1
            n_correct += 1
        else:
            if is_one:
                failure_one.append(grid_input)
            else:
                failure_mul.append(grid_input)

    print('{}/{}'.format(n_correct,batch_size*(b+1)))
    if g == 2:
        print('S: {}/{}'.format(n_correct_one,n_one))
        print('M: {}/{}'.format(n_correct_mul,n_mul))
with open('_success_one.pkl','wb') as f:
    pickle.dump(success_one, f)
with open('_success_mul.pkl','wb') as f:
    pickle.dump(success_mul, f)
with open('_failure_one.pkl','wb') as f:
    pickle.dump(failure_one, f)
with open('_failure_mul.pkl','wb') as f:
    pickle.dump(failure_mul, f)

