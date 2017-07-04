import numpy as np
import tensorflow as tf
import mf
import sys
import pickle
import sudoku

np.set_printoptions(precision=3, suppress=True)

with open(sys.argv[1],'rb') as f:
    w,u = pickle.load(f)

pad_zeros = True
zero_val = True

g = 3
n_iter = 300
n = p = g**2

if zero_val or pad_zeros:
    p += 1
if pad_zeros:
    n *= 2


print(w)
print(w.shape)
print(u.shape)
print("####")

ann_w = np.zeros((n_iter,n,n,p,p))
ann_u = np.zeros((n_iter,n,n,p))
for i in range(n_iter):
    prog = i/float(n_iter)
    coef = (1 / (1+np.exp(-prog*5)) - 0.5)*2*10 + 0.5
    ann_w[i] = w*coef
    ann_u[i] = u*coef



weights = tf.convert_to_tensor(ann_w, dtype=tf.float32)
unary = tf.convert_to_tensor(ann_u, dtype=tf.float32)

m = mf.MeanField(n,n,p)
t,e,c = m.build_model(weights, unary, n_iter, damping=0.5)
q = tf.nn.softmax(-t)


test_grid = np.array([[4,2,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]])
test_grid = np.array([[5,3,0,0,7,0,0,0,0],
                      [6,0,0,1,9,5,0,0,0],
                      [0,9,8,0,0,0,0,6,0],
                      [8,0,0,0,6,0,0,0,3],
                      [4,0,0,8,0,3,0,0,1],
                      [7,0,0,0,2,0,0,0,6],
                      [0,6,0,0,0,0,2,8,0],
                      [0,0,0,0,0,9,0,0,5],
                      [0,0,0,0,8,0,0,7,9]])

otest_grid = np.array([[5,3,4,6,7,8,9,1,2],
                      [6,7,2,1,9,5,3,4,8],
                      [1,9,8,3,4,2,5,6,7],
                      [8,5,9,7,6,1,4,2,3],
                      [4,2,6,8,5,3,7,9,1],
                      [7,1,3,9,2,4,8,5,6],
                      [9,6,1,5,3,7,2,8,4],
                      [2,8,7,4,1,9,6,3,5],
                      [0,0,0,0,0,0,0,0,0]])
        
clip = sudoku.grid_to_clip(sudoku.expand_matrix(sudoku.to_prob(test_grid,p),g,p))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
q_,t_,e_ = sess.run([q,t,e], feed_dict={c: [clip]})
q_  = q_[0]
print(q_)
print(t_)
print(e_)
print(q.shape)
print(sudoku.infer_grid_probabilities(sudoku.reduce_matrix(q_,g,p)))
print(sudoku.infer_grid(sudoku.reduce_matrix(q_,g,p)))
sess.close()
n_modes = 11

weights = tf.convert_to_tensor(w, dtype=tf.float32)
unary = tf.convert_to_tensor(u, dtype=tf.float32)
mmmf = mf.MultiModalMeanField(n,n,p,weights,unary,n_iter,damping=0.5)
sess = tf.Session()
for _ in range(n_modes):
    mmmf.iteration(sess)


q_values    = mmmf.get_q_mf_values()
mode_probs  = mmmf.get_modes_energy()

parameters = {
                mmmf._theta_clip: np.array(mmmf._modes),
                mmmf._T: 1/15.
             }
q,prob = sess.run([q_values,mode_probs], feed_dict=parameters)
pr,q_ = min(zip(prob,q))
print(sudoku.infer_grid_probabilities(sudoku.reduce_matrix(q_,g,p)))
print(sudoku.infer_grid(sudoku.reduce_matrix(q_,g,p)))
print(pr)
print(mmmf._modesT)
sess.close()
