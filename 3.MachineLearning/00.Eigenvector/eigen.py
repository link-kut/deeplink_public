import numpy as np
import numpy.linalg as linalg
from numpy.linalg import matrix_rank
#Compute the eigenvalues and right eigenvectors of a square array.

np.set_printoptions(precision=3)

def printEigen(P):
    W, V = linalg.eig(P)

    for i in range(len(W)):
        print("Eigenvalue:", W[i], "- The Corresponding Eigenvector:", V[:, i])


def LI_vecs(dim, M):
    LI=[M[0]]
    for i in range(dim):
        tmp=[]
        for r in LI:
            tmp.append(r)
        tmp.append(M[i])                #set tmp=LI+[M[i]]
        if matrix_rank(tmp)>len(LI):    #test if M[i] is linearly independent from all (row) vectors in LI
            LI.append(M[i])             #note that matrix_rank does not need to take in a square matrix
    return LI                           #return set of linearly independent (row) vectors


P = np.array([[2, -12], [1, -5]])
print(P)
printEigen(P)
print(LI_vecs(2, P))

print("-----------")

P = np.array([[1, -2], [3, -4]])
print(P)
printEigen(P)
print(LI_vecs(2, P))

print("-----------")

P = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 2]])
print(P)
printEigen(P)
print(LI_vecs(3, P))

print("-----------")

P = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
print(P)
printEigen(P)
print(LI_vecs(3, P))

print("-----------")

P = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
print(P)
printEigen(P)
print(LI_vecs(3, P))

print("-----------")

