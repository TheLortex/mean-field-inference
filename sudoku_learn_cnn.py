from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import pickle
import sys
import argparse
from tqdm import tqdm
import os
import itertools
import setproctitle
import sudoku
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3,suppress=True)

version = 2.5

parser = argparse.ArgumentParser()
parser.add_argument('--boardSz', type=int, default=2)
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--out', type=str, default='latest.pkl')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--decay', type=int, default=5)
parser.add_argument('--nepoch', type=int, default=50)
args = parser.parse_args()

setproctitle.setproctitle('sudoku_learning {} -> {}'.format(args.dataset, args.out))


print('loading dataset')
with open(args.dataset,'rb') as f:
    dataset_X, dataset_Y = pickle.load(f)
print('Dataset loaded')
n_samples,_,_,_ = dataset_X.shape

g = args.boardSz
n = g**2
p = g**2

# remove zeroval
inputs = dataset_X[:,:,:,1:p+1]
labels = dataset_Y[:,:,:,1:p+1]
tf_samples = tf.placeholder(tf.float32,[args.bs, n, n, p])
tf_ground_truth = tf.placeholder(tf.float32,[args.bs, n, n, p])

x_one_hot = tf.expand_dims(tf.eye(n), axis=1)
x_one_hot = tf.tile(x_one_hot, [1, n, 1])
y_one_hot = tf.expand_dims(tf.eye(n), axis=0)
y_one_hot = tf.tile(y_one_hot, [n, 1, 1])

location_feature = tf.concat([x_one_hot, y_one_hot], 2) # of shape (n,n,2n)
input_location = tf.tile(tf.expand_dims(location_feature,0), [args.bs, 1, 1, 1])

batch_input = tf.concat([tf_samples, input_location], axis=3) # of shape (bs,n,n,p+2n)

convolutions = [(n,n,256),(n,n,128),(n,n,64),(n,n,64),(n,n,p)]

current_input = batch_input
current_depth = p+2*n
i = 0
for w,h,k in convolutions:
    with tf.name_scope('conv_{}'.format(i)):
        conv_layer = tf.Variable(tf.random_normal((w,h,current_depth,k), stddev=np.sqrt(2)/np.sqrt(current_depth)),name="conv_params_{}".format(i))
        bias_relu = tf.Variable(tf.random_normal([k], stddev=np.sqrt(2)/np.sqrt(current_depth)),name="bias_params_{}".format(i))
        output_conv = tf.nn.conv2d(current_input, conv_layer, [1, 1, 1, 1], "SAME")
        current_input = tf.nn.elu(output_conv+bias_relu) if i < len(convolutions)-1 else output_conv + bias_relu
        current_depth = k
        i += 1

with tf.name_scope('final_layer'):
    output = tf.nn.softmax(current_input)


with tf.name_scope('loss'):
    p_times_q = output * tf_ground_truth
    log_likelihood = tf.reduce_sum(tf.log(tf.reduce_sum(p_times_q, 3)+0.0000001), axis=(1,2))

    loss = -tf.reduce_mean(log_likelihood)

batch_size = args.bs
    
step = tf.Variable(0, trainable=False)
rate = args.lr*tf.pow(0.7,tf.cast(tf.div(step, (n_samples//batch_size)*args.decay), tf.float32)) # decrease learning rate every 5 epoch

train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=step)


print('tf graph is built')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#writer = tf.summary.FileWriter(logdir='output_summary', graph=tf.get_default_graph())
#writer.flush()

#print('tf graph saved')
saver = tf.train.Saver()

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

evol = [[],[],[]]

for i in tqdm(range(args.nepoch),desc='epoch'):
    for b in tqdm(range(n_samples//batch_size),desc='batch'):

        parameters = {
                    tf_samples: inputs[b*batch_size:(b+1)*batch_size],
                    tf_ground_truth: labels[b*batch_size:(b+1)*batch_size]
                }

        loss_value,_ = sess.run([loss,train_op], feed_dict=parameters)
        evol[0].append(loss_value)
        evol[1].append(0)
        evol[2].append(0)
    saver.save(sess,args.out) 
    with open(args.out+'.lrn','wb') as f:
        pickle.dump(evol,f)
print('done')

n_correct = 0
n = 0

#n_samples = batch_size

ex = [  [0, 1, 2, 0],
        [0, 2, 1, 0],
        [1, 3, 4, 2],
        [2, 4, 3, 1]]
    
#inputs[:batch_size] = sudoku.to_prob(np.array(ex),5)[:,:,1:]
for b in range(n_samples//batch_size):
    parameters = {tf_samples: inputs[b*batch_size:(b+1)*batch_size]}
    output_values = sess.run(output, feed_dict=parameters)
    for grid_input, grid_output in zip(inputs[b*batch_size:(b+1)*batch_size],output_values):
        grid = sudoku.infer_grid(grid_output)
        correct = sudoku.is_correct(1+grid,g)
        #if args.explain:
        #    print('###')
        #    print(grid)
        #    print(sudoku.infer_grid_probabilities(grid_output))
        #    print(correct)
        if n < 10:
            print(sudoku.infer_grid(grid_input))
            print(grid_output)
            print(grid)
            print('####')
        n += 1
        if correct:
            n_correct += 1
    print('{}/{}       '.format(n_correct,n), end='\r')
print()

