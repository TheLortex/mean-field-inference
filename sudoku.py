import numpy as np
import itertools

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
    n_samples,m,_,p = grid.shape
    clipping = np.zeros((n_samples,2,m,m,p))
    clipping[:,0,:,:,:] = -50 #min clip
    clipping[:,1,:,:,:] =  50 #max clip
    for s in range(n_samples):
        for x in range(m):
            for y in range(m):
                for k in range(p):
                    if grid[s,x,y,k] == 1:
                        for k_2 in range(p):
                            if k != k_2:
                                clipping[s,0,x,y,k_2] = 50
                        clipping[s,1,x,y,k] = -50
    return clipping

def reduce_matrix(graph,g,p):
    res = np.zeros((g**2,g**2,p))
    for x in range(g):
        for y in range(g):
            res[g*x:g*(x+1),g*y:g*(y+1)] = graph[g*(2*x):g*(2*x+1),g*(2*y):g*(2*y+1)]
    return res


def expand_matrix(dataset,g,p):
    dataset = np.array(dataset)
    n_samples,_,_,_ = dataset.shape
    target = np.zeros((n_samples,2*(g**2),2*(g**2),p))
    target[:,:,0] = 1
    for x in range(g):
        for y in range(g):
                target[:,g*(2*x):g*(2*x+1),g*(2*y):g*(2*y+1)] = dataset[:,g*x:g*(x+1),g*y:g*(y+1)]
    return target

def is_correct(grid,g):
    if np.any(grid == 0):
        return False

    ok_l = [sum(line) == sum(set(line)) for line in grid]
    ok_c = [sum(col) == sum(set(col)) for col in np.transpose(grid)]
    squares = []
    for i in range(g):
        for j in range(g):
            square = np.concatenate([row[g*j:g*(j+1)] for row in grid[g*i:g*(i+1)]])
            squares.append(square)
    ok_sq = [sum(square) == sum(set(square)) for square in squares]
    return np.all(ok_l+ok_c+ok_sq)
    

def values_ok(x,y,grid,g):
    ok = set(range(1,g**2+1))
    for exp in range(g**2):
        ok.discard(grid[x][exp])
        ok.discard(grid[exp][y])
    sq_x = g*(x//g)
    sq_y = g*(y//g)
    for px in range(g):
        for py in range(g):
            ok.discard(grid[sq_x+px][sq_y+py])
    return list(ok)

def n_solutions_grid(grid,g):
    for x in range(g**2):
        for y in range(g**2):
            if grid[x][y] == 0:
                ns = 0
                for t in values_ok(x,y,grid,g):
                    grid[x][y] = t
                    ns += n_solutions_grid(grid,g)
                grid[x][y] = 0
                return ns
    return 1
