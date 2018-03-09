import coarsening
import tensorflow as tf
import numpy as np
import scipy as spy

# A=[[(0,1),0.3],
#    [(0,2),0.4],
#    [(0,3),0.3],
#    [(1,2),0.3],
#    [(1,3),0.3],
#    [(1,4),0.4],
#    [(2,0),0.4],
#    [(2,3),0.3],
#    [(2,4),0.3],
#    [(3,0),0.3],
#    [(3,1),0.4],
#    [(3,4),0.3],
#    [(4,0),0.3],
#    [(4,2),0.4],
#    [(4,3),0.3]]
A=spy.sparse.rand(5,5,0.3,'coo',None)
print(A)

graphs,perm=coarsening.coarsen(A,1,False)
print('graphs:')
print(graphs)
print('perm:')
print(perm)