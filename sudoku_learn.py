from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import mf
import pickle
import sys
import argparse
from tqdm import tqdm
import os
import itertools
import setproctitle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3,suppress=True)

version = 2.5

parser = argparse.ArgumentParser()
parser.add_argument('--boardSz', type=int, default=2)
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--out', type=str, default='latest.pkl')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--lognmodes', type=int, default=0)
parser.add_argument('--mfiter', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--nepoch', type=int, default=50)
parser.add_argument('--annealing', default=False, action='store_true')
parser.add_argument('--old', default=False, action='store_true')
parser.add_argument('--invbyperm', default=False, action='store_true')
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--h', type=int, default=0)
parser.add_argument('--nopad', default=False, action='store_true')
parser.add_argument('--decay', type=int, default=5)

args = parser.parse_args()

setproctitle.setproctitle('sudoku_learning {} -> {}'.format(args.dataset, args.out))


print('Starting experiment!')
print('Board size:',args.boardSz)
print('Dataset:',args.dataset)
print('Output in:',args.out)
print('Input weights:',args.input)
print('Number of modes:',2**args.lognmodes)
print('Mean field iters:',args.mfiter)
print('Learning rate:',args.lr)
print('Batch size:', args.bs)
print('Number of epochs:', args.nepoch)
print('Annealing:', args.annealing)
print('Using old loss:', args.old)
print('Invariant by permutation:',args.invbyperm)
print('k:',args.k)
print('h:',args.h)
print('padding:',not(args.nopad))
print('decay:', args.decay)
k = args.k
h = args.h
pad_zeros = not(args.nopad)
zero_val = True

meanfield_iters = args.mfiter
batch_size = args.bs
n_epoch = args.nepoch
learning_rate = args.lr
n_modes = args.lognmodes

print('loading dataset')
with open(args.dataset,'rb') as f:
    dataset_X, dataset_Y = pickle.load(f)
print('Dataset loaded')
n_samples,_,_,_ = dataset_X.shape

g = args.boardSz
n = g**2
p = g**2

if zero_val or pad_zeros:
    p += 1
if pad_zeros:
    n *= 2


#weights_np = np.random.normal(0,np.sqrt(2/(n*n)),size=(k,n,n,p,p))
#weights_np[:,0,0,:,:] = 0
#unary_np = np.random.normal(0,np.sqrt(2/(n*n)),size=(n,n,p))
weights_np = np.zeros((k,n,n,p,p))
unary_np = np.zeros((n,n,p))
annealing_np = None
FNN_np = None

# TODO: Fix preload for new version.
if args.input != '':
    with open(args.input,'rb') as f:
        model = pickle.load(f)
        weights_np = model['links']
        unary_np = model['unary']
        annealing_np = model['annealing']

        FNN_np = model['FNN']
        print("Loaded from file")
        print(weights_np)
        print(unary_np)
        print(annealing_np)
        print("Loaded from file")
        assert meanfield_iters == annealing_np.shape[0]
        assert weights_np.shape == (k,n,n,p,p)
        assert unary_np.shape == (n,n,p)

if annealing_np is None:
    annealing_np = np.zeros((meanfield_iters),dtype=np.float32)

m = mf.MeanField(n, n, p)
print('meanfield initialized')
def expand(dataset):
    n_samples,_,_,_ = dataset.shape
    target = np.zeros((n_samples,n,n,p))
    target[:,:,:,0] = 1
    for x in range(g):
        for y in range(g):
            target[:,g*(2*x):g*(2*x+1),g*(2*y):g*(2*y+1)] = dataset[:,g*x:g*(x+1),g*y:g*(y+1)]
    return target

def grid_to_clip(grid):
    n_samples,n,_,p = grid.shape
    res = np.zeros((n_samples,2,n,n,p))
    for s in range(n_samples):
        res[s] = m.theta_clip_nothing()
    res[:,1,:,:,:] = np.where(grid == 1, -50, res[:,1,:,:,:])
    return res

print('preprocessing data')

if pad_zeros:
    inputs = grid_to_clip(expand(dataset_X))
    labels = expand(dataset_Y)
elif not zero_val:
    inputs = grid_to_clip(dataset_X[:,:,:,1:p+1])
    labels = dataset_Y[:,:,:,1:p+1]
else:
    inputs = grid_to_clip(dataset_X)
    labels = dataset_Y
print('preprocessed data')
print(dataset_X.shape)


tf_samples = tf.placeholder(tf.float32,[None, n, n, p])
links = tf.Variable(tf.convert_to_tensor(weights_np, dtype=tf.float32),name="links")
links_sym = links+tf.transpose(links, [0, 1, 2, 4, 3]) # pairwise weights are symmetric
unary = tf.Variable(tf.convert_to_tensor(unary_np, dtype=tf.float32),name="unary")
unary_sym = unary
annealing = tf.Variable(tf.convert_to_tensor(annealing_np, dtype=tf.float32),name="annealing")


L1      = tf.random_normal((2*n,h), stddev=np.sqrt(1/n))
L1_b    = tf.random_normal([h], stddev=np.sqrt(1/n))
L2      = tf.random_normal((h,k), stddev=np.sqrt(2/(h+1)))
L2_b    = tf.random_normal([k], stddev=np.sqrt(2/(h+1)))

if not(FNN_np is None):
    L1, L1_b, L2, L2_b = FNN_np

FNN = (tf.Variable(L1,name="L1"), tf.Variable(L1_b,name="L1_b"), tf.Variable(L2,name="L2"), tf.Variable(L2_b,name="L2_b"))
L1, L1_b, L2, L2_b = FNN

mmmf = mf.BatchedMultiModalMeanField(n, n, p, batch_size, links_sym, unary_sym, tf.exp(annealing), meanfield_iters, k=k, h=h, FNN=FNN)

with tf.name_scope('computing_q'):
    q_mf = tf.nn.softmax(-mmmf._theta_mf) #  sample, each mode, q values.
    # tf_sample: (s, n, n, p)
    # q_mf: (s*m, n, n, p)
    q_mf = tf.reshape(q_mf, (batch_size,-1,n,n,p))
    # q_mf: (s,m,n,n,p)
with tf.name_scope('count'):
    number_of_modes = tf.shape(q_mf)[1]
    number_of_samples = tf.shape(tf_samples)[0]

with tf.name_scope('log_likelihood'):
    tf_samples_stack = tf.tile(tf.expand_dims(tf_samples, 1), [1, number_of_modes, 1, 1, 1], name="tf_samples_stack") # of shape s, m, n, n, p
    prod = tf_samples_stack*q_mf # broadcast magic -> s,m,n,n,p
    # => log_lh: (s, m)
    log_lh_samples = tf.reduce_sum(tf.log(tf.reduce_sum(prod,4)+0.0000001), (2,3), name="log_lh_samples")
    #log likelihood for each sample each mode.
    sample_mode = tf.argmax(log_lh_samples, axis=1, name="sample_mode")
    # shape is [number_of_samples]
    log_lh = tf.reduce_max(log_lh_samples, axis=1, name="log_lh")

with tf.name_scope('energy'):
    energies = mmmf.get_modes_energy()
    modes_prob = mmmf.get_modes_probability()
    mean_energy = tf.reduce_sum(energies*tf.stop_gradient(modes_prob), name="mean_energy",axis=1)
    energy_sample_mode_gathered = tf.gather(tf.transpose(energies), sample_mode, name="energy_sample_mode_gathered")
    energy_sample_mode = tf.reduce_sum(energy_sample_mode_gathered*tf.eye(batch_size), axis=0, name="energy_sample_mode")

with tf.name_scope('loss_function'):
    to_maximize_old = tf.reduce_mean(log_lh - energy_sample_mode + mean_energy,name="old_loss")
    mean_log_lh = tf.reduce_mean(log_lh,name="mean_log_lh")
    log_prob_sample_mode_gathered = tf.gather(tf.transpose(modes_prob), sample_mode, name="log_prob_sample_mode_gathered")
    log_prob_sample_mode = tf.reduce_sum(log_prob_sample_mode_gathered*tf.eye(batch_size), axis=0, name="log_prob_sample_mode")
    to_maximize = tf.reduce_mean(log_lh + log_prob_sample_mode,name="loss")
    if args.old:
        to_maximize, to_maximize_old = to_maximize_old, to_maximize

variables = [links, unary]
if args.annealing:
    variables.append(annealing)
if h > 0:
    variables.extend([L1, L1_b, L2, L2_b])

with tf.name_scope('gradient'):
    gradient_total = tf.gradients(to_maximize, variables, name="gradient")
#    gradient_total_old = tf.gradients(to_maximize_old, variables, name="old_gradient")
    c = np.tile(np.expand_dims(m.weights_clip_nothing(), 1), [1, k, 1, 1, 1, 1])

    gradient_total[0] = tf.clip_by_value(gradient_total[0], c[0], c[1])
    gradient_norm = sum([tf.reduce_sum(tf.abs(gradient_total[i])) for i in range(len(gradient_total))])

with tf.name_scope('update'):
    update_rate = learning_rate*batch_size/float(n_samples)

    updates = []
    print("Printing learned variables")
    for idx, elem in enumerate(variables):
        print(idx, elem)
        updates.append(tf.assign(elem, tf.check_numerics(elem + update_rate*gradient_total[idx],"gradient_check_{}".format(idx)),name="update_{}".format(idx)))


step = tf.Variable(0, trainable=False)
rate = args.lr*tf.pow(0.7,tf.cast(tf.div(step, (n_samples//batch_size)*args.decay), tf.float32)) # decrease learning rate every 5 epoch

train_op = tf.train.AdamOptimizer(rate).minimize(-to_maximize, global_step=step, var_list=variables)


print('tf graph is built')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#writer = tf.summary.FileWriter(logdir='output_summary', graph=tf.get_default_graph())
#writer.flush()

#print('tf graph saved')

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

evol = [[],[],[]]

for i in tqdm(range(n_epoch),desc='epoch'):
    for b in tqdm(range(n_samples//batch_size),desc='batch'):
        #print(np.expand_dims(np.array(inputs[b*batch_size:(b+1)*batch_size]),axis=1).shape)
        mmmf.reset_all(np.expand_dims(np.array(inputs[b*batch_size:(b+1)*batch_size]),axis=1))
        for _ in range(n_modes):
            mmmf.iteration(sess)

        parameters = {
                    mmmf._theta_clip: np.reshape(mmmf._modes,(-1,2,n,n,p)),
                    mmmf._T: 1,
                    tf_samples: labels[b*batch_size:(b+1)*batch_size]
                }

        operations = [mean_log_lh,to_maximize,gradient_norm]
        operations.append(train_op)
        loglh,loss,gradient,_ = sess.run(operations, feed_dict=parameters)
        evol[0].append(loglh)
        evol[1].append(gradient)
        evol[2].append(loss)


    with open(args.out,'wb') as f:
        data = {}
        data['links'], data['unary'], data['annealing'], data['FNN'] = sess.run([links, unary, annealing, FNN])
        data['args'] = args
        data['version'] = version
        pickle.dump(data, f)
    with open(args.out+'.lrn','wb') as f:
        pickle.dump(evol,f)
print('done')
