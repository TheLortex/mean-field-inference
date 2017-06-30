import numpy as np
import tensorflow as tf

import mf

n = 8
p = 2

np.set_printoptions(precision=2, suppress=True)

print('MF TEST')
target = np.zeros([50, n, n, p, p])
for nx, ny in [(1,0),(-1,0),(0,1),(0,-1)]:
    for v1 in range(2):
        for v2 in range(2):
            if v1 == v2:
                target[:,nx,ny,v1,v2] = 1
            else:
                target[:,nx,ny,v1,v2] = -1

weights = tf.convert_to_tensor(target, dtype=tf.float32)
unary = tf.zeros([50, n, n, p])

m = mf.MeanField(n,n,p)
theta,energy,clip_ph = m.build_model(0.5*weights, unary, 50, damping=0.20)
q = tf.nn.softmax(-theta)

sess = tf.Session()


c = m.theta_clip_nothing()
c[0,0,0,0] = 50
c[1,0,0,1] = -50

print(sess.run(q[:,:,:,0], feed_dict={clip_ph: np.array([c])}))

print('MMMF Test')

mmmf = mf.MultiModalMeanField(sess,n,n,p,weights[0],unary[0])
mmmf.iteration()
parameters = {mmmf._theta_clip: np.array(mmmf._modes),
                mmmf._T: 2}
q_values = sess.run(mmmf.get_q_mf_values(), feed_dict=parameters)[:,:,:,0]
print(mmmf.get_modes_clip())
print(q_values)
sess.close()

print("LEARNING TEST")
links = tf.Variable(tf.zeros([n, n, p, p]))
unary = tf.Variable(tf.zeros([n, n, p]))
tf_samples = tf.placeholder(tf.float32, [None, n, n, p], name="samples")


samples = np.zeros(shape=(2,n,n,p))
for i in range(n):
    for j in range(n):
        if (i+j)%2 == 0:
            samples[0,i,j,0] = 1
            samples[1,i,j,1] = 1
        else:
            samples[0,i,j,1] = 1
            samples[1,i,j,0] = 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())
mmmf = mf.MultiModalMeanField(sess, n, n, p, links, unary)

q_mf = mmmf.get_q_mf_values() # for each mode, q values.
# tf_sample: (s, n, n, p)
# q_mf: (m, n, n, p)
# > prod: (m, s, n, n, p)
number_of_modes = tf.shape(q_mf)[0]
number_of_samples = tf.shape(tf_samples)[0]
tf_samples_stack = tf.tile(tf.expand_dims(tf_samples, 0), [number_of_modes, 1, 1, 1, 1]) # of shape m, s, n, n, p
tf_mode_stack = tf.tile(tf.expand_dims(q_mf, 1), [1, number_of_samples, 1, 1, 1]) # of shape m, s, n, n, p
prod = tf_samples_stack*tf_mode_stack

# => log_lh: (m, s)
log_lh_samples = tf.reduce_sum(tf.log(tf.reduce_sum(prod,4)), (2,3))
#log likelihood for each mode, each sample.
sample_mode = tf.argmax(log_lh_samples, axis=0)
# shape is [number_of_samples]
log_lh = tf.reduce_max(log_lh_samples, axis=0)
#log likelihood for each sample.
#we want to maximize that.

# sum of
energies = mmmf.get_modes_energy()
modes_prob = mmmf.get_modes_probability()
mean_energy = tf.reduce_mean(energies*tf.stop_gradient(modes_prob))
energy_sample_mode = tf.gather(energies, sample_mode)

to_maximize = log_lh - energy_sample_mode + mean_energy

gradient_total = tf.gradients(to_maximize, [links, unary])
c = m.weights_clip_nothing()
gradient_total[0] = tf.clip_by_value(gradient_total[0], c[0], c[1])
update_weights = tf.assign(links, links + 0.1*gradient_total[0])
update_unary = tf.assign(unary, unary + 0.1*gradient_total[1])


for i in range(30):
    print("##########################")
    mmmf.reset_all()
    for _ in range(1): # find maximum 2 modes
        ok, maxq = mmmf.iteration()
        print(maxq)
        if not ok:
            break

    parameters = {
                    mmmf._theta_clip: mmmf._modes,
                    mmmf._T: 1,
                    tf_samples: samples
                }
    s,w,u = sess.run([sample_mode, update_weights, update_unary], feed_dict=parameters)
print(sess.run(links))
mmmf = mf.MultiModalMeanField(sess,n,n,p,weights[0],unary[0])
mmmf.iteration()
parameters = {mmmf._theta_clip: np.array(mmmf._modes),
                mmmf._T: 2}

print(sess.run(mmmf.get_q_mf_values(), feed_dict=parameters)[:,:,:,0])
s
