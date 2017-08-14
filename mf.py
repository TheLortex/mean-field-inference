from __future__ import print_function

import numpy as np
import tensorflow as tf 
# A Mean Field inference layer with a fixed number of iterations. 
class MeanField():
    # Input: 
    #  -n,m,p: n*m picture, random variable space of size p. 
    #  -k: number of filters
    #  -h: hidden layer for filter selection.
    #  -n_iter: number of iterations during the learning phase. 
    #  -weights: Tensor of shape (k,n,m,p,p) convolutional energy between points defined for k filters
    #  -unary: Tensor of shape (n,m,p) unary terms of pixels. 

    #  These Tensors can be of shape (n_iter,n,m,..), this way they define the 
    #  values at each iteration.
    
    def __init__ (self, n, m, p, k=1, h=0, theta_std=0.1):    
        self._n     = n
        self._m     = m
        self._p     = p
        self._k     = k
        self._h     = h
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
    def __compute_theta_star(self, links, unary, theta, filter_selection):
        with tf.name_scope('theta_star'): 
            # links is of shape (k,n,n,p,p)
            # should have shape (n,n,p,k*p)
            links_reshaped = tf.reshape(tf.transpose(links,[1,2,3,4,0]), (self._n, self._n, self._p, self._k*self._p)) # hoping the magic works
            self._links_reshaped = links_reshaped
            q = tf.nn.softmax(-theta) # (bs,n,n,p)
            E_reshaped = self.__compute_circular_convolution(q, links_reshaped) # (bs,n,n,k*p)
            self._E_reshaped = E_reshaped
            self._q = q
            E_per_filter = tf.reshape(E_reshaped, (-1, self._n, self._n, self._p, self._k)) # of shape (bs,n,n,p,k)
            E_per_filter = tf.transpose(E_per_filter,[4,0,1,2,3])
            self._E_per_filter = E_per_filter
            #filter selection is of shape (n,n,k)
            filter_sel_transposed = tf.transpose(filter_selection, perm=[2,0,1]) # of shape (k,n,n)
            filter_sel_prepared = tf.expand_dims(filter_sel_transposed, axis=1)
            filter_sel_prepared = tf.expand_dims(filter_sel_prepared, axis=-1) # of shape (k,1,n,n,1), ready for broadcasting. 
            E_selected = E_per_filter*filter_sel_prepared

            return tf.reduce_sum(E_selected, axis=0) + unary

    def __update_rule(self, links, unary, theta, d, filter_selection):
        assert (0 < d < 1)
        with tf.name_scope('update') as scope:
            theta_star = self.__compute_theta_star(links, unary, theta, filter_selection)
            res = d*theta_star + (1-d)*theta
        return res

    def build_model (self, weights, unary, n_iter, damping=0.05, FNN=(0,0,0,1), placeholder=None):
        # weights of shape (k,n,n,p,p)
        # unary of shape (n,n,p)
        # FNN (filter neural network) of shape ((2*n,h), (h), (h,k), (k))
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
        (L1, L1b, L2, L2b) = FNN


        if placeholder is None:
            placeholder = tf.placeholder(tf.float32, shape=[None, 2, self._n, self._m, self._p], name="theta_clip")
        self._theta_clip = placeholder
        with tf.name_scope('theta'):
            self._theta = tf.random_normal(tf.shape(self._theta_clip[:,0,:,:,:], name="theta_shape"), stddev=0.1, name="initial_theta")


        x_one_hot = tf.expand_dims(tf.eye(self._n), axis=1) # of shape (n,1,n) 
        x_one_hot = tf.tile(x_one_hot, [1, self._n, 1]) # of shape (n,n,n) 1st dim is taken into account
        y_one_hot = tf.expand_dims(tf.eye(self._n), axis=0)
        y_one_hot = tf.tile(y_one_hot, [self._n, 1, 1]) # of shape (n,n,n) 2nd dim is taken into account

        result = tf.concat([x_one_hot, y_one_hot], 2) # of shape (n,n,2n) - x then y coordinate encoding.
        if self._h > 0: 
            tmp = tf.tensordot(result,L1,1)
            self._hidden_layer = tf.nn.tanh(tmp+L1b)                               # of shape (n,n,h)
            
            tmp = tf.tensordot(self._hidden_layer,L2,1) 
            self._filter_selection = tf.nn.softmax(tmp+L2b) # of shape (n,n,k) i.e. filter 
                                                                             # composition for each coordinate, 
        else:                                                                # allowing the definition of more complex CRFs.
            self._filter_selection = tf.nn.softmax(tf.zeros((self._n,self._n,self._k))+L2b)

        # MF-inference loop unroll
        self._weights = weights
        self._unary = unary
        self._theta_mf = self._theta
        with tf.name_scope('mf_inference'):
            for i in range(n_iter):
                with tf.name_scope('mf_loop') as scope:
                    new_theta = self.__update_rule(weights[i], unary[i], self._theta_mf, damping, self._filter_selection)
                    self._theta_mf = tf.clip_by_value(
                                    new_theta, 
                                    self._theta_clip[:,0],
                                    self._theta_clip[:,1])

        q = tf.nn.softmax(-self._theta_mf)
        with tf.name_scope('energy'):
            #E = self.__compute_circular_convolution(q, self._weights[0])
            #self._energy = tf.reduce_sum((E+self._unary[0])*q,axis=(1,2,3)) 
            links_reshaped = tf.reshape(self._weights[0], (self._n, self._n, self._p, self._k*self._p)) # hoping the magic works
            
            E_reshaped = self.__compute_circular_convolution(q, links_reshaped) # (bs,n,n,k*p)
            E_per_filter = tf.reshape(E_reshaped, (self._k, -1, self._n, self._n, self._p)) # of shape (k,bs,n,n,p)

            #filter selection is of shape (n,n,k)
            filter_sel_transposed = tf.transpose(self._filter_selection, perm=[2,0,1]) # of shape (k,n,n)
            filter_sel_prepared = tf.expand_dims(filter_sel_transposed, axis=1)
            filter_sel_prepared = tf.expand_dims(filter_sel_prepared, axis=-1) # of shape (k,1,n,n,1), ready for broadcasting. 
            E_selected = E_per_filter*filter_sel_prepared
            self._energy = tf.reduce_sum((tf.reduce_sum(E_selected, axis=0) + self._unary[0])*q, axis=(1,2,3))


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



class BatchedMultiModalMeanField():
    def __init__(self, n, m, p, bs, links, unary, annealing, n_iter, damping=0.5, k=1, h=0, FNN=(0,0,0,1)):
       
        # links is of shape k, n, n, p, p
        # needs to add annealing dimension. 
        
        if annealing is None:
            self._links = [links]*n_iter
            self._unary = [unary]*n_iter
        else:
            ann = annealing
            for _ in range(3):
                ann = tf.expand_dims(ann, -1)
            ann_unary = ann                     # of shape (n_iter, 1, 1, 1)
            for _ in range(2):
                ann = tf.expand_dims(ann, -1)
            ann_links = ann                     # of shape (n_iter, 1, 1, 1, 1, 1)
         
            self._links     = ann_links*tf.tile(tf.expand_dims(links, 0), [n_iter, 1, 1, 1, 1, 1])
            self._unary     = ann_unary*tf.tile(tf.expand_dims(unary, 0), [n_iter, 1, 1, 1])
        
        self._T         = tf.placeholder(tf.float32,name="Temperature")

        self._mf = MeanField(n, m, p, k, h)
        self._theta_mf, energy, self._theta_clip = self._mf.build_model(self._links/self._T, self._unary/self._T, n_iter, damping, FNN)
        self._energy = tf.reshape(energy, (bs, -1))
        self._modes_probabilities = tf.nn.softmax(-self._energy)
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
        n_modes_total = self._nmodes*self._bs
        remaining = n_modes_total
        results_q = np.zeros((self._n, self._n, self._p, self._bs, self._nmodes))
        results_entropy = np.zeros((self._n, self._n, self._bs, self._nmodes))
        unfinished = np.array([[True]*self._nmodes for _ in range(self._bs)])

        parameters = {
                        self._theta_clip: np.reshape(self._modes,(self._nmodes*self._bs,2,self._n,self._n,self._p)),
                        self._T: T
                     }
        q = np.reshape(sess.run(self._q_mf, feed_dict=parameters),(self._bs,self._nmodes,self._n,self._n,self._p))
        q = np.transpose(q, [2, 3, 4, 0, 1]) 
        unclipped_values = np.all((self._modes[:,:,0] + self._modes[:,:,1]) != -100, axis=-1)
        unclipped_values = np.transpose(unclipped_values, [2,3,0,1])
        entropy0 = (-np.sum(q*np.log2(q+0.0000001), axis=2)) < 0.3*np.log2(self._p) # Initial entropy. 
        n_iter = 0
        while True:
            if not(np.any(unfinished)):
                return np.transpose(results_q, [3,4,0,1,2]), np.transpose(results_entropy, [2,3,0,1])

            parameters = {
                            self._theta_clip: np.reshape(self._modes,(self._nmodes*self._bs,2,self._n,self._n,self._p)),
                            self._T: T
                         }
            q = np.reshape(sess.run(self._q_mf, feed_dict=parameters),(self._bs,self._nmodes,self._n,self._n,self._p))
            q = np.transpose(q, [2, 3, 4, 0, 1]) #n,n,p,bs,nmodes

            entropy = -np.sum(q*np.log2(q+0.0000001), axis=2) #n,n,bs,nmodes
            


            phase_transition = np.logical_and(entropy > 0.7*np.log2(self._p), entropy0) # of shape n,n,bs,nmodes
            has_phase_transition = np.any(phase_transition, axis=(0,1)) # of shape bs,nmodes
            
            should_update = np.logical_and(has_phase_transition, unfinished) # find modes who have their first phase transition
            if np.sum(should_update) == 0:
                n_iter += 1
                if n_iter > 20:
                    print("too many iterations now stopping at T=",T,'unfinished=',np.sum(unfinished))
                    should_update = unfinished

            unfinished = np.logical_xor(should_update, unfinished) # clear bits
            results_q = np.where(should_update, q, results_q)
            results_entropy = np.where(should_update, entropy, results_entropy)
            T *= 1.2


    def iteration(self, session):
        q, entropy = self.find_phase_transition(session, 1/5.)
        newmodes = np.zeros((self._bs,self._nmodes,2,self._n,self._n,self._p))
        for i in range(self._bs):
            for j in range(self._nmodes):
                idx = np.argmax(entropy[i,j])
                x,y = np.unravel_index(idx, entropy[i,j].shape)
                k = np.argmax(q[i,j,x,y])
                cur_mode        = self._modes[i,j]
                new_mode        = cur_mode.copy()

                cur_mode[0,x,y,k] = -50
                cur_mode[1,x,y,k] = -50

                new_mode[0,x,y,k] =  50
                new_mode[1,x,y,k] =  50

                newmodes[i,j] = new_mode
        self._modes = np.concatenate((self._modes, newmodes), axis=1)
        self._nmodes *= 2
    
    def get_energy_values(self):
        return self._energy

    def get_modes(self):
        return self._modes

    def get_q_mf_values(self):
        return self._q_mf

    def get_modes_probability(self):
        return self._modes_probabilities

    def get_modes_energy(self):
        return self._energy





class TensorflowizedBatchedMultiModalMeanField():
    def __init__(self, n, m, p, bs, links, unary, annealing, n_iter, damping=0.5, k=1, h=0, FNN=(0,0,0,1)):
       
        # links is of shape k, n, n, p, p
        # needs to add annealing dimension. 
        ann = annealing
        for _ in range(3):
            ann = tf.expand_dims(ann, -1)
        ann_unary = ann                     # of shape (n_iter, 1, 1, 1)
        for _ in range(2):
            ann = tf.expand_dims(ann, -1)
        ann_links = ann                     # of shape (n_iter, 1, 1, 1, 1, 1)
        self._links     = ann_links*tf.tile(tf.expand_dims(links, 0), [n_iter, 1, 1, 1, 1, 1])
        self._unary     = ann_unary*tf.tile(tf.expand_dims(unary, 0), [n_iter, 1, 1, 1])
        
        self._T         = tf.placeholder(tf.float32,name="Temperature")

        self._mf = MeanField(n, m, p, k, h)
        
        self._theta_mf, energy, self._theta_clip = self._mf.build_model(self._links/self._T, self._unary/self._T, n_iter, damping, FNN)

        self._energy = tf.reshape(energy, (bs, -1))
        self._modes_probabilities = tf.nn.softmax(-self._energy)
        self._q_mf = tf.nn.softmax(-self._theta_mf)
        #self._modes = np.array([[self._mf.theta_clip_nothing()] for _ in range(bs)])
        #self._nmodes = 1
        self._bs = bs
        self._n = n
        self._p = p
        self._n_iter = n_iter
        self._FNN = FNN
        self._damping = damping
    # provide a batched mode list
    #def reset_all(self, initial_modes=None):
    #    if initial_modes is None:
    #        initial_modes = np.array([[self._mf.theta_clip_nothing()] for _ in range(bs)])
    #    self._modes = initial_modes
    #    self._nmodes = 1

    def find_phase_transition(self, currentModes, nmodes, T):
        n_modes_total = nmodes*self._bs

        theta_clip = tf.reshape(currentModes, [nmodes*self._bs,2,self._n,self._n,self._p])
        theta_mf0, _, _ = self._mf.build_model(self._links/T, self._unary/T, self._n_iter, self._damping, self._FNN, theta_clip)

        q0 = tf.reshape(tf.nn.softmax(-theta_mf0),[self._bs,nmodes,self._n,self._n,self._p])
        entropy0 = (-tf.reduce_sum(q0*tf.log(q0+0.0000001),axis=4)) < 0.3*tf.log(tf.constant(10.0))


        results_q = tf.zeros((self._bs, nmodes, self._n, self._n, self._p))
        unfinished = tf.convert_to_tensor(np.array([[True]*nmodes for _ in range(self._bs)]), dtype=tf.bool)
        n_iter = tf.constant(0)

        def cond(unfinished, n_iter, T, results_q):
            return tf.logical_and(tf.reduce_any(unfinished), tf.less(n_iter,20))

        def body(unfinished, n_iter, T, results_q):
            theta_mfi, _, _ = self._mf.build_model(self._links/T, self._unary/T, self._n_iter, self._damping, self._FNN, theta_clip)
            qi = tf.reshape(tf.nn.softmax(-theta_mfi),[self._bs,nmodes,self._n,self._n,self._p])
            entropyi = -tf.reduce_sum(qi*tf.log(qi+0.0000001), axis=4)

            phase_transition = tf.logical_and(entropy0, tf.greater(entropyi, 0.7*tf.log(tf.constant(10.0)))) # of shape bs,nmodes,n,n
            has_phase_transition = tf.reduce_any(phase_transition, axis=[2,3]) # of shape bs,nmodes
            should_update = tf.logical_and(has_phase_transition, unfinished)
            should_update_n_n = tf.expand_dims(tf.expand_dims(should_update,-1),-1)
            should_update_n_n_p = tf.expand_dims(should_update_n_n,-1)


            n_iter_next = tf.cond(tf.reduce_any(should_update), lambda: n_iter, lambda: tf.add(n_iter,1))
            T_next = T * 1.2
            unfinished_next = tf.logical_xor(should_update, unfinished)
            results_q_next = tf.where(tf.tile(should_update_n_n_p, [1,1,self._n,self._n,self._p]), qi, results_q)
            
            return unfinished_next, n_iter_next, T_next, results_q_next
        # [50,1], [1], [1], [bs,nmodes,n,n,p]
        _,_,_,q_final = tf.while_loop(cond,body,(unfinished,n_iter,T,results_q))
        return q_final

    def iteration(self, currentModes, nmodes): 
        q = self.find_phase_transition(currentModes, nmodes, tf.constant(1/5.))
        entropy = tf.reduce_sum(q*tf.log(q+0.0000001), axis=4)
        
        newmodes = currentModes

        entropy_flattened = tf.reshape(entropy,[self._bs, nmodes, -1])
        q_flattened = tf.reshape(q,[self._bs, nmodes, -1, self._p])

        max_entropy_pos = tf.argmax(entropy_flattened, axis=2)

        q_selected = tf.gather(tf.transpose(q_flattened,[2,0,1,3]),max_entropy_pos)
        # of shape bs*nmodes * bs*nmodes*p
        q_reshaped  = tf.reshape(q_selected,[self._bs*nmodes, self._bs*nmodes, self._p])
        identity_select = tf.expand_dims(tf.eye(self._bs*nmodes),2)
        
        q_flat_maxentropy = tf.reduce_sum(q_reshaped*identity_select,axis=1) # of shape (bs*nmodes,p)
        q_maxentropy = tf.reshape(q_flat_maxentropy, [self._bs,nmodes,self._p]) # of shape (bs, nmodes)
        maxq_maxentropy = tf.argmax(q_maxentropy, axis=2) # of shape (bs, nmodes)

        bs_list = tf.expand_dims(tf.cumsum(tf.ones([self._bs],dtype=tf.int64), exclusive=True),1)
        modes_list = tf.expand_dims(tf.cumsum(tf.ones([nmodes],dtype=tf.int64), exclusive=True),0)

        bs_mtrx = tf.tile(bs_list, [1, nmodes])
        modes_mtrx = tf.tile(modes_list, [self._bs, 1])



        n0 = tf.div(max_entropy_pos, self._n)
        n1 = tf.mod(max_entropy_pos, self._n)


        update_coordinates = tf.stack([bs_mtrx, modes_mtrx,n0, n1, maxq_maxentropy], axis=2) # of shape (bs, nmodes, 5)

        ref = currentModes # of shape (bs, nmodes, 2, n, n, p), P = 6
        indices = update_coordinates # of shape (bs, nmodes, 6) Q = 3
        # K = P
        # updates of rank Q-1, shape (bs, nmodes)
        updates = 100*tf.ones([self._bs, nmodes])
        
        modesClamp = tf.scatter_nd(update_coordinates,updates,[self._bs, nmodes, self._n, self._n, self._p])
        modesClamp = tf.expand_dims(modesClamp, 2)
        # of shape (bs, nmodes, 1, n, n, p)
        zeroClamp = tf.zeros([self._bs, nmodes, 1, self._n, self._n, self._p])
        
        modesClampMax = tf.concat([zeroClamp, modesClamp], axis=2)
        modesClampMin = tf.concat([modesClamp, zeroClamp], axis=2)
        # of shape (bs, nmodes, 2, n, n, p)

        modesClampPlus = currentModes + modesClampMin # set min to 50 for clamped variables
        modesClampMinus = currentModes - modesClampMax # set max to -50 for clamped variables

        newModes = tf.concat([modesClampPlus, modesClampMinus], axis=1)
        return newModes     

    def get_energy_values(self):
        return self._energy

    def get_modes(self):
        return self._modes

    def get_q_mf_values(self):
        return self._q_mf

    def get_modes_probability(self):
        return self._modes_probabilities

    def get_modes_energy(self):
        return self._energy


