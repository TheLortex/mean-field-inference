from __future__ import print_function
import os
for nmodes in [0]:
    print(' nmodes:',nmodes)
    for boardSz in [2]:
        print('  boardSz:',boardSz)
        for kind in ['mul']:
            print('   kind:', kind)
            for decay in [4,8,16,32]:
                print('    decay:',decay)
                for i in range(10):
                    name = 'experiment10/{}_{}_{}_{}_{}.pkl'.format(nmodes,boardSz,kind,decay,i)
                    if not(os.path.isfile(name)):
                        nsq = boardSz**2
                        input_dataset = 'dataset_{}x{}_{}.pkl'.format(nsq,nsq,kind)
                        command = 'python sudoku_learn.py --boardSz {} --nopad --k 5 --h 10 --bs 200 --nepoch 50 --decay {} --lognmodes {} --dataset {} --out {}'.format(boardSz,decay,nmodes,input_dataset,name)
                        os.system(command)
