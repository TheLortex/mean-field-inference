import numpy as np
import tensorflow as tf 

# A Mean Field inference layer with a fixed number of iterations. 
class MeanField():
    # Input: 
    #  -n,m,p: n*m picture, random variable space of size p. 
    #  -n_iter: number of iterations during the learning phase. 
    #  -weights: Tensor of shape (n,m,p,p) convolutional energy between points
    #  -unary: Tensor of shape (n,m,p) unary terms of pixels. 
    #  These Tensors can be of shape (n_iter,n,m,..), this way they define the 
    #  values at each iteration.
    
    def __init__ (self, n, m, p, theta_std=0.1):    
        self._n     = n
        self._m     = m
        self._p     = p
        with tf.name_scope('theta'):
            self._theta_clip = tf.placeholder(tf.float32, shape=[None, 2, n, m, p], name="theta_clip")
            self._theta = tf.random_normal(tf.shape(self._theta_clip[:,0,:,:,:], name="theta_shape"), stddev=theta_std, name="initial_theta")

    # image shape (bs, n, m, p) | filter shape (n, m, p, p)
    # outputs convolution of shape (bs, n, m, p)
    def __compute_circular_convolution (self, image, filtr):
        with tf.name_scope('convolution'):
            image_1 = tf.concat(2*[image], axis=1)
            image_2 = tf.concat(2*[image_1], axis=2) # shape (bs, 2*n, 2*m, p)
            image_3 = tf.slice(image_2, 
                        [0, 0, 0, 0], 
                        [-1, 2*self._n-1, 2*self._m-1,-1])
            strides = [1, 1, 1, 1]
            padding = "VALID"
            return tf.nn.conv2d(image_3, filtr, strides, padding)

    # takes theta of shape (bs, n, m, p)
    def __compute_theta_star(self, links, unary, theta):
        with tf.name_scope('theta_star'):
            q = tf.nn.softmax(-theta)
            E = self.__compute_circular_convolution(q, links)
            return E + unary

    def __update_rule(self, links, unary, theta, d):
        assert (0 < d < 1)
        with tf.name_scope('update') as scope:
            theta_star = self.__compute_theta_star(links, unary, theta)
            res = d*theta_star + (1-d)*theta
        return res

    def build_model (self, weights, unary, n_iter, damping=0.05):
        #shape_w = tf.shape(weights)
        #shape_u = tf.shape(unary)
                                                                          
        #if shape_w.shape == [4]:
        #    weights = tf.stack(n_iter*[weights])
        #    shape_w = tf.shape(weights)
        # 
        #if shape_u.shape == [3]:
        #    unary = tf.stack(n_iter*[unary])
        #    shape_u = tf.shape(unary)
                                                                          
        # perform shape checks
        #assert (len(shape_w) == 5 and len(shape_u) == 4)
        #assert (shape_w[0] == n_iter == shape_u[0])
        #assert (shape_w[1] == n == shape_u[1])
        #assert (shape_w[2] == m == shape_u[2])
        #assert (shape_w[3] == shape_w[4] == p == shape_u[3])
        

        # MF-inference loop unroll
        self._weights = weights
        self._unary = unary
        theta = self._theta

        with tf.name_scope('mf_loop') as scope:
            for i in range(n_iter):
                new_theta = self.__update_rule(weights[i], unary[i], theta, damping)
                theta = tf.clip_by_value(
                            new_theta, 
                            self._theta_clip[:,0],
                            self._theta_clip[:,1])
            self._theta_mf = theta
            q = tf.nn.softmax(-self._theta_mf)
            tf.summary.histogram('q',q)
            with tf.name_scope('energy'):
                E = self.__compute_circular_convolution(q, self._weights[0])
                self._energy = tf.reduce_sum((E+self._unary[0])*q,axis=(1,2,3)) 
                tf.summary.histogram('energy',self._energy)        

        return self._theta_mf, self._energy, self._theta_clip

    def get_energy (self):
        return self._energy

    def get_inference (self):
        return self._theta_mf

    def get_clipping_ph (self):
        return self._theta_clip

    def theta_clip_nothing (self):
        clip = np.zeros((2, self._n,self._m,self._p))
        clip[0,:,:,:] = -50
        clip[1,:,:,:] =  50
        return clip

    def weights_clip_nothing (self):
        clip = np.zeros((2, self._n, self._m, self._p, self._p))
        clip[0,:,:,:,:] = -50
        clip[1,:,:,:,:] =  50
        clip[:,0,0,:,:] = 0
        return clip




class MultiModalMeanField():
    def __init__(self, n, m, p, links, unary, annealing, n_iter, damping=0.5):
        for _ in range(4):
            annealing = tf.expand_dims(annealing)
        annealing = tf.tile(annealing, [1, n, m, p, p])
        self._links     = tf.stack(n_iter*[links])*annealing
        self._unary     = tf.stack(n_iter*[unary])*annealing
        self._T         = tf.placeholder(tf.float32,name="Temperature")

        self._mf = MeanField(n, m, p)
        self._theta_mf, self._energy, self._theta_clip = self._mf.build_model(self._links/self._T, self._unary/self._T, n_iter, damping)
        self._modes_probabilities = tf.nn.softmax(self._energy)
        self._q_mf = tf.nn.softmax(-self._theta_mf)
        self._modes = [self._mf.theta_clip_nothing()]
        self._modesT = [0]

    def reset_all(self, initial_modes=None):
        if initial_modes is None:
            initial_modes = [self._mf.theta_clip_nothing()]
        self._modes = initial_modes
        self._modesT = [0]

    def find_phase_transition(self, sess, T):
        parameters = {
                        self._theta_clip: np.array(self._modes),
                        self._T: T
                    }
        q = sess.run(self._q_mf, feed_dict=parameters)
        entropy0 = -np.sum(q*np.log2(q+0.0000001), axis=3)
        zerosure = entropy0 < 0.3*np.log2(10) # TODO: 10 -> scale
        while True:
            T *= 1.2
            parameters = {
                            self._theta_clip: np.array(self._modes),
                            self._T: T
                         }
            q = sess.run(self._q_mf, feed_dict=parameters)
            entropy = -np.sum(q*np.log2(q+0.0000001), axis=3)

            chk = np.logical_and(entropy > 0.7*np.log2(10), zerosure)
            if np.any(chk):
                return q, chk, T

    def iteration(self, session):
        q, chk, T = self.find_phase_transition(session, 1/10.)
        
        idx = np.argmax(chk)
        m,x,y = np.unravel_index(idx, chk.shape)
        k = np.argmax(q[m,x,y])
        print(m,x,y,k,T,q[m,x,y])


        self._modesT[m] = T
        self._modesT.append(T)

        cur_mode        = self._modes[m]
        new_mode        = cur_mode.copy()

        cur_mode[0,x,y,k] = -50
        cur_mode[1,x,y,k] = -50

        new_mode[0,x,y,k] =  50
        new_mode[1,x,y,k] =  50

        self._modes.append(new_mode)
        return True, q[m,x,y,k]

    def get_modes(self):
        return self._modes

    def get_q_mf_values(self):
        return self._q_mf

    def get_modes_probability(self):
        return self._modes_probabilities

    def get_modes_energy(self):
        return self._energy


class BatchedMultiModalMeanField():
    def __init__(self, n, m, p, bs, links, unary, annealing, n_iter, damping=0.5):
        
        self._links     = tf.transpose(tf.stack(n_iter*[links],-1)*annealing, [4,0,1,2,3])
        self._unary     = tf.transpose(tf.stack(n_iter*[unary],-1)*annealing, [3,0,1,2])
        print(self._links.shape)
        print(links.shape)
        self._T         = tf.placeholder(tf.float32,name="Temperature")

        self._mf = MeanField(n, m, p)
        self._theta_mf, energy, self._theta_clip = self._mf.build_model(self._links/self._T, self._unary/self._T, n_iter, damping)
        self._energy = tf.reshape(energy, (bs, -1))
        self._modes_probabilities = tf.nn.softmax(self._energy)
        self._q_mf = tf.nn.softmax(-self._theta_mf)
        self._modes = np.array([[self._mf.theta_clip_nothing()] for _ in range(bs)])
        self._nmodes = 1
        self._bs = bs
        self._n = n
        self._p = p

    # provide a batched mode list
    def reset_all(self, initial_modes=None):
        if initial_modes is None:
            initial_modes = np.array([[self._mf.theta_clip_nothing()] for _ in range(bs)])
        self._modes = initial_modes
        self._nmodes = 1

    def find_phase_transition(self, sess, T):
        while True:
            parameters = {
                            self._theta_clip: np.reshape(self._modes,(self._nmodes*self._bs,2,self._n,self._n,self._p)),
                            self._T: T
                         }
            q = np.reshape(sess.run(self._q_mf, feed_dict=parameters),(self._bs,self._nmodes,self._n,self._n,self._p))
            entropy = -np.sum(q*np.log2(q+0.0000001), axis=4)
            return q,entropy
            # todo: fix it
            if np.max(entropy) > 0.7 or T >= 0.1*(2**8): #8 iterations max
                return q, entropy 
            T *= 2

    def iteration(self, session):
        q, entropy = self.find_phase_transition(session, 1)
        newmodes = np.zeros((self._bs,1,2,self._n,self._n,self._p))
        for i in range(self._bs):
            idx = np.argmax(entropy[i])
            #print(q[i].shape, entropy[i].shape, idx)
            m,x,y = np.unravel_index(idx, entropy[i].shape)
            k = np.argmax(q[i,m,x,y])
            #print(self._modes[i,m,:,x,y])
            #print(i,m,x,y,k,q[i,m,x,y,k])

            cur_mode        = self._modes[i,m]
            new_mode        = cur_mode.copy()

            cur_mode[0,x,y,k] = -50
            cur_mode[1,x,y,k] = -50

            new_mode[0,x,y,k] =  50
            new_mode[1,x,y,k] =  50

            newmodes[i,0] = new_mode
        self._modes = np.concatenate((self._modes, newmodes), axis=1)
        self._nmodes += 1

    def get_modes(self):
        return self._modes

    def get_q_mf_values(self):
        return self._q_mf

    def get_modes_probability(self):
        return self._modes_probabilities

    def get_modes_energy(self):
        return self._energy



