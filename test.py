import numpy as np
import tensorflow as tf
import mf
import sys
import pickle
import sudoku

np.set_printoptions(precision=3, suppress=True)

with open(sys.argv[1],'rb') as f:
    w,u,a = pickle.load(f)

pad_zeros = True
zero_val = True

g = 3
n_iter = a.shape[0]
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
    ann_w[i] = w*a[i]
    ann_u[i] = u*a[i]



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
        
clip = sudoku.grid_to_clip(sudoku.expand_matrix([sudoku.to_prob(test_grid,p)],g,p))

merged = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


writer = tf.summary.FileWriter('output_summary', graph=tf.get_default_graph())
summary, q_,t_,e_ = sess.run([merged,q,t,e], feed_dict={c: clip})
writer.add_summary(summary,0)
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
mmmf = mf.BatchedMultiModalMeanField(n,n,p,1,weights,unary,a,n_iter,damping=0.5)
mmmf.reset_all(np.array([clip]))
sess = tf.Session()
for _ in range(n_modes):
    mmmf.iteration(sess)


q_values    = mmmf.get_q_mf_values()
mode_probs  = mmmf.get_modes_energy()

parameters = {
                mmmf._theta_clip: np.reshape(mmmf._modes,(-1,2,n,n,p)),
                mmmf._T: 1/5.
             }
q,prob = sess.run([q_values,mode_probs], feed_dict=parameters)
print(q.shape)
print(prob.shape)
ok = False
pr,q_ = min(zip(prob[0],q))
print(q_.shape)
print(sudoku.infer_grid_probabilities(sudoku.reduce_matrix(q_,g,p)))
grid = (sudoku.infer_grid(sudoku.reduce_matrix(q_,g,p)))
print(grid)
print(pr)
print(sudoku.is_correct(grid,g))
nok = 0
for q_,pr in zip(q,prob[0]):
    grid = sudoku.infer_grid(sudoku.reduce_matrix(q_,g,p))
    ok_ = sudoku.is_correct(grid,g)
    if ok_:
        print('ok:',pr)
        print(grid)
        nok += 1
    ok = ok or ok_
print(nok)
sess.close()
