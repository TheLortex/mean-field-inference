from __future__ import print_function
import os
import judge
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

experiences = [# boardSz,nprelearn,nlearn,lognmodes,k,h,annealing,nopadding,dataset_difficulty
                # the difficulty of simple learning on difficult dataset, even with padding
                (2,0,50,0,1,0,False,False,"one"),
                (2,0,50,0,1,0,False,False,"mul"), 
                (2,0,50,0,1,0,False,False,"mul_hard"),
                # multi-modal mean-field overcome this problem:
                (2,0,50,2,1,0,False,False,"one"),
                (2,0,50,2,1,0,False,False,"mul"),
                (2,0,50,4,1,0,False,False,"mul_hard"),
                # but removing the padding fucks everything:
                (2,0,50,0,1,0,False,True,"one"),
                (2,0,50,2,1,0,False,True,"mul"),
                # so let's add multiple filters possibility:
                (2,0,50,0,10,20,False,True,"one"),
                # and with multi-modal on difficult datasets:
                (2,0,50,2,10,20,False,True,"mul"),
                # with pre-learning to get even better results:
                (2,40,10,2,10,20,False,True,"mul"),
                # On 9x9:
                (3,40,10,2,10,20,False,True,"mul"),
                (3,0,50,0,10,20,False,True,"mul"),
                (3,0,50,2,10,20,False,True,"mul"),
                (3,40,10,0,10,20,False,True,"one"),
                (2,40,10,2,10,20,True,True,"mul"),
                (3,40,25,2,10,20,False,True,"mul"),
                (3,0,65,0,10,20,False,True,"mul")

            ]



for i,(boardSz, nprelearn, nepoch, lognmodes, k, h, annealing, nopad, dataset) in enumerate(experiences):
    name = 'fullexp/' + ("_".join([str(x) for x in list(experiences[i])]))
    if not(os.path.isfile(name)):
        print(name)
        nsq = boardSz**2
        input_dataset = 'dataset/dataset_{}x{}_{}.pkl'.format(nsq,nsq,dataset)
        command = 'python sudoku_learn.py --bs 200 --decay 16'
        command += ' --boardSz {}'.format(boardSz)
        command += ' --nprelearn {}'.format(nprelearn)
        command += ' --nepoch {}'.format(nepoch)
        command += ' --lognmodes {}'.format(lognmodes)
        command += ' --k {}'.format(k)
        command += ' --h {}'.format(h)
        command += ' --dataset {}'.format(input_dataset)
        command += ' --out {}'.format(name)
        if annealing:
            command += ' --annealing'
        if nopad:
            command += ' --nopad'
        print(command)
        os.system(command)

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]


res_exp = open('results.txt','a')
for i,(boardSz, nprelearn, nepoch, lognmodes, k, h, annealing, nopad, dataset) in enumerate(experiences):
    name = 'fullexp/' + ("_".join([str(x) for x in list(experiences[i])]))
    if not(os.path.isfile(name+'.png')): 
        learning = pickle.load(open(name+'.lrn','rb'))
        plt.clf()
        plt.plot(moving_average(learning[0],10000/200))
        plt.savefig(name+'.png')

        nsq = boardSz ** 2
        dataset = 'dataset/dataset_{}x{}_mul_2.pkl'.format(nsq,nsq)
        for lognmodes in [0,2]:
            args = lambda: None
            args.dataset = dataset
            args.input = name
            args.boardSz = boardSz
            args.bs = 200
            args.lognmodes = lognmodes
            args.n = -1
            args.explain = False
            n_correct, n_samples, n_correct_tot, n_tot = judge.judge(args)
            p_correct = 100* (n_correct / float(n_samples))
            p_explored = 100* (n_correct_tot / float(n_tot))
            res_exp.write('{} : {} | {}\n'.format(name, p_correct, p_explored))
            
close(res_exp)
