from __future__ import print_function
import os
for nmodes in [2]:
    print(' nmodes:',nmodes)
    for boardSz in [2,3]:
        print('  boardSz:',boardSz)
        for kind in ['one','mul']:
            print('   kind:', kind)
            for annealing in [True]:
                print('    annealing:',annealing)
                name = 'experiment2/{}_{}_{}_{}.pkl'.format(nmodes,boardSz,kind,annealing)
                if not(os.path.isfile(name)):
                    nsq = boardSz**2
                    input_dataset = 'dataset_{}x{}_{}.pkl'.format(nsq,nsq,kind)
                    command = 'python sudoku_learn.py --boardSz {} --lognmodes {} --dataset {} --out {}'.format(boardSz,nmodes,input_dataset,name)
                    if annealing:
                        command = command + ' --annealing'
                    os.system(command)
