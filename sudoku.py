import numpy as np

def infer_grid(prob_grid):
    return np.argmax(prob_grid, axis=2)

def infer_grid_probabilities(prob_grid):
    return np.max(prob_grid, axis=2)

def to_prob(grid,p):
    m,_ = grid.shape
    prob_grid = np.zeros((m,m,p))
    for x in range(m):
        for y in range(m):
            if grid[x,y] != 0:
                prob_grid[x,y,grid[x,y]] = 1
    return prob_grid

def grid_to_clip(grid):
    m,_,p = grid.shape
    clipping = np.zeros((2,m,m,p))
    clipping[0,:,:,:] = -50 #min clip
    clipping[1,:,:,:] =  50 #max clip

    for x in range(m):
        for y in range(m):
            for k in range(p):
                if grid[x,y,k] == 1:
                    for k_2 in range(p):
                        if k != k_2:
                            clipping[0,x,y,k_2] = 50
                    clipping[1,x,y,k] = -50
    return clipping

def reduce_matrix(graph,g,p):
    res = np.zeros((g**2,g**2,p))
    for x in range(g):
        for y in range(g):
            res[g*x:g*(x+1),g*y:g*(y+1)] = graph[g*(2*x):g*(2*x+1),g*(2*y):g*(2*y+1)]
    return res


def expand_matrix(dataset,g,p):
    _,_,_ = dataset.shape
    target = np.zeros((2*(g**2),2*(g**2),p))
    target[:,:,0] = 1
    for x in range(g):
        for y in range(g):
            target[g*(2*x):g*(2*x+1),g*(2*y):g*(2*y+1)] = dataset[g*x:g*(x+1),g*y:g*(y+1)]
    return target

