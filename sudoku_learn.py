import numpy as np
import tensorflow as tf
import mf 
import pickle
import sys
np.set_printoptions(precision=3,suppress=True)


pad_zeros = True
zero_val = True

batch_size = 50
n_epoch = 500
learning_rate = 0.05
n_modes = 0

with open(sys.argv[1],'rb') as f:
    dataset_X, dataset_Y = pickle.load(f)

n_samples,_,_,_ = dataset_X.shape

g = 3
n = g**2
p = g**2

if zero_val or pad_zeros:
    p += 1
if pad_zeros:
    n *= 2


weights_np = np.zeros((n,n,p,p),dtype=np.float32)
unary_np = np.zeros((n,n,p),dtype=np.float32)

if len(sys.argv) >= 4:
    with open(sys.argv[3],'rb') as f:
        weights_np, unary_np = pickle.load(f)
        print("Loaded from file")
        print(weights_np)
        print(unary_np)
        print("Loaded from file")
        assert weights_np.shape == (n,n,p,p)
        assert unary_np.shape == (n,n,p)


m = mf.MeanField(n, n, p)

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
        for x in range(n):
            for y in range(n):
                for k in range(p):
                    if grid[s,x,y,k] == 1:
                        res[s,1,x,y,k] = -50
    return res

with open(sys.argv[1],'rb') as f:
    dataset_X, dataset_Y = pickle.load(f)
#n_samples,_,_,_ = dataset_Y.shape


if pad_zeros:
    inputs = grid_to_clip(expand(dataset_X))
    labels = expand(dataset_Y)
elif not zero_val:
    inputs = grid_to_clip(dataset_X[:,:,:,1:p+1])
    labels = dataset_Y[:,:,:,1:p+1]
else:
    inputs = grid_to_clip(dataset_X)
    labels = dataset_Y

print(dataset_X.shape)

tf_samples = tf.placeholder(tf.float32,[None, n, n, p])
links = tf.Variable(tf.convert_to_tensor(weights_np))
links_sym = links+tf.transpose(links, [0, 1, 3, 2]) # pairwise weights are symmetric
unary = tf.Variable(tf.convert_to_tensor(unary_np))
mmmf = mf.BatchedMultiModalMeanField(n, n, p, batch_size, links_sym, unary)

q_mf = mmmf.get_q_mf_values() # for each sample, each mode, q values.
# tf_sample: (s, n, n, p)
# q_mf: (s*m, n, n, p)
q_mf = tf.reshape(q_mf, (batch_size,-1,n,n,p))

number_of_modes = tf.shape(q_mf)[1]
number_of_samples = tf.shape(tf_samples)[0]
tf_samples_stack = tf.tile(tf.expand_dims(tf_samples, 1), [1, number_of_modes, 1, 1, 1]) # of shape s, m, n, n, p
prod = tf_samples_stack*q_mf

# => log_lh: (s, m)
log_lh_samples = tf.reduce_sum(tf.log(tf.reduce_sum(prod,4)+0.0000001), (2,3))
#log likelihood for each sample each mode.
sample_mode = tf.argmax(log_lh_samples, axis=1)
# shape is [number_of_samples]
log_lh = tf.reduce_max(log_lh_samples, axis=1)

# sum of
energies = mmmf.get_modes_energy()
modes_prob = mmmf.get_modes_probability()
mean_energy = tf.reduce_sum(energies*modes_prob, axis=1)
energy_sample_mode = tf.gather(tf.transpose(energies), sample_mode)

to_maximize = tf.reduce_mean(log_lh - energy_sample_mode + mean_energy)

gradient_total = tf.gradients(to_maximize, [links, unary])
c = m.weights_clip_nothing()
gradient_total[0] = tf.clip_by_value(gradient_total[0], c[0], c[1])


update_rate = learning_rate*batch_size/float(n_samples)

update_weights = tf.assign(links, tf.check_numerics(links + update_rate*gradient_total[0],"weights"))
update_unary = tf.assign(unary, tf.check_numerics(unary + update_rate*gradient_total[1],"unary"))

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


for i in range(n_epoch):
    print("##############",i,"##############")
    for b in range(n_samples/batch_size):
        if b%10 == 0:
            print(b)
        #print(np.expand_dims(np.array(inputs[b*batch_size:(b+1)*batch_size]),axis=1).shape)
        mmmf.reset_all(np.expand_dims(np.array(inputs[b*batch_size:(b+1)*batch_size]),axis=1))
        for _ in range(n_modes): 
            mmmf.iteration(sess)

        parameters = {
                    mmmf._theta_clip: np.reshape(mmmf._modes,(-1,2,n,n,p)),
                    mmmf._T: 1,
                    tf_samples: labels[b*batch_size:(b+1)*batch_size]
                }
        aa,a,b,c,tm,g_w,s,w,u = sess.run([prod, log_lh, energy_sample_mode, mean_energy, to_maximize, gradient_total,sample_mode, update_weights, update_unary], feed_dict=parameters)
    #print(aa[0])
    #print(g_w[0])
    #print(g_w[1])
    #print(a,b,c)
#        print(b)
#        print(w)
#        print(u)

        
    with open('{}_{}.pkl'.format(sys.argv[2],i),'wb') as f:
        pickle.dump(sess.run([links, unary]), f)

print(sess.run(links))
mmmf = mf.MultiModalMeanField(n,n,p,links_sym,unary)
mmmf.iteration(sess)
parameters = {mmmf._theta_clip: np.array(mmmf._modes),
                mmmf._T: 1}

print(sess.run(mmmf.get_q_mf_values(), feed_dict=parameters))
