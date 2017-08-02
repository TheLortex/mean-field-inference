# On Mean-Field inference and machine learning: the example of Sudoku. 
Work done during the Summer 2017 internship at the CVLab - EPFL. 

* create.py: create a sudoku dataset, you can specify the size of the board, the number of examples and the difficulty of the dataset.
* discr.py: this script tests the discriminative power of a given CRF (not updated to version 2.0)
* experience.py: launches a bunch of experiments (not updated)
* judge.py: evaluates the quality of a learned CRF against a dataset, specifying the number of modes (in case of multi-modal mean-field). 
* learning.py: some MF samples (not updated)
* mf.py: the core MF/MMMF inference library 
* sudoku_learn.py: uses the MMMF inference to learn the game of sudoku using a multi-modal gradient method.
* sudoku.py: some utility library in order to handle sudoku dataset. 
