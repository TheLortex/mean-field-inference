import numpy as np
import tensorflow as tf
import mf 

g = 4
n_iter = 300

p = 1+g**2
n = 2*(g**2)

exp = 2

def from_one_hot(vec):
    return np.argmax(vec)

def to_grid(one_hot_grid):
    m,_,_ = one_hot_grid.shape
    grid = np.zeros((m,m))
    for x in range(m):
        for y in range(m):
            grid[x,y] = from_one_hot(one_hot_grid[x,y])
    return grid

def to_grid_prob(one_hot_grid):
    return np.max(one_hot_grid, axis=2)

target = np.zeros([n_iter, n, n, p, p])
for x in range(1,n):
    for v1 in range(p):
        for v2 in range(p):
            if v1 == v2 and v1 != 0:
                target[:,x,0,v1,v2] = 5
                target[:,0,x,v1,v2] = 5
            else:
                target[:,x,0,v1,v2] = -1
                target[:,0,x,v1,v2] = -1
for x,y in [(2*g,0),(2*g,2*g),(0,2*g)]:
    target[:,x,y,:,0] = 5
    target[:,x,y,0,:] = 5
    target[:,x,y,0,0] = 0

for x in range(-g+1,g):
    for y in range(-g+1,g):
        print(x,y)
        for v1 in range(1,p):
            for v2 in range(1,p):
                if v1 == v2:
                    target[:,x,y,v1,v2] = 5
                else:
                    target[:,x,y,v1,v2] = -1

unary = np.zeros([n_iter, n, n, p])
for x in range(n):
    for y in range(n):
        if x%(2*g) >= g or y%(2*g) >= g:
            unary[:,x,y,0] = -15
        else:
            unary[:,x,y,0] = 15

weights = tf.convert_to_tensor(target, dtype=tf.float32)
unary = tf.convert_to_tensor(unary, dtype=tf.float32)
m = mf.MeanField(n,n,p,theta_std=1)
theta,energy,clip_ph = m.build_model(weights/1.5, unary, n_iter, damping=0.1)
q = tf.nn.softmax(-theta)


mmmf = mf.MultiModalMeanField(n,n,p,weights[0],unary[0])
q_modes = mmmf.get_q_mf_values()
e_modes = mmmf.get_modes_energy()

sess = tf.Session()

grids = [np.zeros((g**2,g**2), dtype=np.int32)]
#grids[0] = [[5,3,0,0,7,0,0,0,0],
#            [6,0,0,1,9,5,0,0,0],
#            [0,9,8,0,0,0,0,6,0],
#            [8,0,0,0,6,0,0,0,3],
#            [4,0,0,8,0,3,0,0,1],
#            [7,0,0,0,2,0,0,0,6],
#            [0,6,0,0,0,0,2,8,0],
#            [0,0,0,4,1,9,0,0,5],
#            [0,0,0,0,8,0,0,7,9]]

#grids[0] = [[4,0,0,0],
#            [0,0,4,0],
#            [0,2,0,0],
#            [0,0,0,1]]


def grid_to_clip(grid):
    base = m.theta_clip_nothing()
    for x in range(g**2):
        for y in range(g**2):
            val = grid[x][y]
            if val != 0:
                x_ = g*(2*(x/g))+x%g
                y_ = g*(2*(y/g))+y%g
                base[1,x_,y_,val] = -50
    return base

clips = []
for grid in grids:
    clips.append(grid_to_clip(grid))
params = {clip_ph: np.array(clips)}
q,t,e = sess.run([q,theta, energy], feed_dict=params)
q=q[0]
t=t[0]
def reduce_graph(graph):
    res = np.zeros((g**2,g**2,p))
    for x in range(g):
        for y in range(g):
            res[g*x:g*(x+1),g*y:g*(y+1)] = graph[g*(2*x):g*(2*x+1),g*(2*y):g*(2*y+1)]
    return res
q_extr = to_grid(reduce_graph(q))
print(reduce_graph(t))
q_probs = to_grid_prob(reduce_graph(q))

print(q_extr.shape)
np.set_printoptions(precision=2, suppress=True)
print(q_probs)
print(q_extr)
print(e)

exit()

for i in range(8):
    print(i)
    mmmf.iteration(sess)


parameters = {mmmf._theta_clip: np.array(mmmf._modes), mmmf._T: 1}
q,e = sess.run([q_modes, e_modes], feed_dict=parameters)


for m in q:
    m_ = reduce_graph(m)
    print(to_grid_prob(m_))
    print(to_grid(m_))
print(np.sum(e < -1055))


sess.close()







