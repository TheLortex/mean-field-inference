from __future__ import print_function
import numpy as np
import tensorflow as tf
import mf 
import pickle
import sys
import argparse
from tqdm import tqdm
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3,suppress=True)


parser = argparse.ArgumentParser()
parser.add_argument('--boardSz', type=int, default=2)
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--out', type=str, default='latest.pkl')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--lognmodes', type=int, default=0)
parser.add_argument('--mfiter', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--nepoch', type=int, default=50)
parser.add_argument('--annealing', default=False, action='store_true')
args = parser.parse_args()
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

pad_zeros = True
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


weights_np = np.zeros((n,n,p,p),dtype=np.float32)
unary_np = np.zeros((n,n,p),dtype=np.float32)
annealing_np = None

if args.input != '':
    with open(args.input,'rb') as f:
        weights_np, unary_np, annealing_np = pickle.load(f)
        print("Loaded from file")
        print(weights_np)
        print(unary_np)
        print(annealing_np)
        print("Loaded from file")
        assert meanfield_iters == annealing_np.shape[0]
        assert weights_np.shape == (n,n,p,p)
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
links = tf.Variable(tf.convert_to_tensor(weights_np))
links_sym = links+tf.transpose(links, [0, 1, 3, 2]) # pairwise weights are symmetric
unary = tf.Variable(tf.convert_to_tensor(unary_np))
annealing = tf.Variable(tf.convert_to_tensor(annealing_np))

mmmf = mf.BatchedMultiModalMeanField(n, n, p, batch_size, links_sym, unary, tf.exp(annealing), meanfield_iters)

q_mf = tf.nn.softmax(-mmmf._theta_mf[-1]) #  sample, each mode, q values.
# tf_sample: (s, n, n, p)
# q_mf: (s*m, n, n, p)
q_mf = tf.reshape(q_mf, (batch_size,-1,n,n,p))

# q_mf: (s,m,n,n,p)

number_of_modes = tf.shape(q_mf)[1]
number_of_samples = tf.shape(tf_samples)[0]
tf_samples_stack = tf.tile(tf.expand_dims(tf_samples, 1), [1, number_of_modes, 1, 1, 1]) # of shape s, m, n, n, p
prod = tf_samples_stack*q_mf # broadcast magic -> s,m,n,n,p

# => log_lh: (s, m)
log_lh_samples = tf.reduce_sum(tf.log(tf.reduce_sum(prod,4)+0.0000001), (2,3))
#log likelihood for each sample each mode.
sample_mode = tf.argmax(log_lh_samples, axis=1)
# shape is [number_of_samples]
log_lh = tf.reduce_max(log_lh_samples, axis=1)

# sum of
energies = mmmf.get_modes_energy()
modes_prob = mmmf.get_modes_probability()
mean_energy = tf.reduce_sum(energies*tf.stop_gradient(modes_prob), axis=1)
energy_sample_mode = tf.gather(tf.transpose(energies), sample_mode)

#to_maximize = tf.reduce_mean(log_lh - energy_sample_mode + mean_energy)
log_sum_exp = tf.reduce_logsumexp(modes_prob)
to_maximize = tf.reduce_mean(log_lh - energy_sample_mode + log_sum_exp)

mean_log_lh = tf.reduce_mean(log_lh)
gradient_total = tf.gradients(to_maximize, [links, unary, annealing])
c = m.weights_clip_nothing()
gradient_total[0] = tf.clip_by_value(gradient_total[0], c[0], c[1])
gradient_norm = tf.reduce_sum(tf.abs(gradient_total[0])) + tf.reduce_sum(tf.abs(gradient_total[1])) + tf.reduce_sum(tf.abs(gradient_total[2]))

update_rate = learning_rate*batch_size/float(n_samples)

update_weights = tf.assign(links, tf.check_numerics(links + update_rate*gradient_total[0],"weights_upd"))
update_unary = tf.assign(unary, tf.check_numerics(unary + update_rate*gradient_total[1],"unary_upt"))
if args.annealing:
    update_annealing = tf.assign(annealing, tf.check_numerics(annealing + update_rate*gradient_total[2],"annealing_upd"))
else:
    update_annealing = tf.assign(annealing, annealing)
print('tf graph is built')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

evol = [[],[],[]]
for i in tqdm(range(n_epoch),desc='epoch'):
    for b in tqdm(range(n_samples/batch_size),desc='batch'):
        #print(np.expand_dims(np.array(inputs[b*batch_size:(b+1)*batch_size]),axis=1).shape)
        mmmf.reset_all(np.expand_dims(np.array(inputs[b*batch_size:(b+1)*batch_size]),axis=1))
        for _ in range(n_modes): 
            mmmf.iteration(sess)

        parameters = {
                    mmmf._theta_clip: np.reshape(mmmf._modes,(-1,2,n,n,p)),
                    mmmf._T: 1,
                    tf_samples: labels[b*batch_size:(b+1)*batch_size]
                }
        loglh,loss,gradient,w,u,a = sess.run([mean_log_lh,to_maximize, gradient_norm, update_weights, update_unary, update_annealing], feed_dict=parameters)
        evol[0].append(loglh)
        evol[1].append(gradient)
        evol[2].append(loss)
        
        
    with open(args.out,'wb') as f:
        pickle.dump(sess.run([links, unary, annealing]), f)
    with open(args.out+'.lrn','wb') as f:
        pickle.dump(evol,f)
print('done')
