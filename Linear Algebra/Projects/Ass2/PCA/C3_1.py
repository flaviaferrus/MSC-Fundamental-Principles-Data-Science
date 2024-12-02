#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:09:41 2022

@author: flaviaferrusmarimon
"""

import pandas as pd
import numpy as np
from numpy.linalg import svd

def loadData(name):
    A = np.loadtxt(name)
    A = A.T
    print(A.shape)
    return A

def computePCA_corr(A, corr = False):
    n = A.shape[1]
    m = A.shape[0]
    r = np.linalg.matrix_rank(A)
    
    print('Number of features: m= ', m) 
    print('Number of samples: n=', n)
    
    if corr:
        print('------------------------------------------')
        print('Computing PCA over the correlation matrix:')
        print('------------------------------------------')
        A_stand = np.zeros(A.shape)
        ## Compute the correlation matrix:
        for i in range(m):
            A_stand[i, :] = (A[i,:] - np.repeat(np.mean(A[i, :]), len(A[i, :])) ) /np.std(A[i,:])

        #print(A_stand)

        ## now we compute the covariance matrix of the standarized matrix:
        Y = 1/np.sqrt(n)* A_stand.T

        ## checking that the matrix corresponds to the correlation matrix:
        #print(np.corrcoef(A))
        #print(np.matmul(Y.T, Y))
        print(np.linalg.norm(np.corrcoef(A) - np.matmul(Y.T, Y)))
    else: 
        print('------------------------------------------')
        print('Computing PCA over the covariance matrix:')
        print('------------------------------------------')
        
        A_centr = np.zeros(A.shape)
        ## Compute the correlation matrix:
        for i in range(m):
            A_centr[i, :] = (A[i,:] - np.repeat(np.mean(A[i, :]), len(A[i, :])) )
        Y = 1/np.sqrt(n-1)* A_centr.T
    
    U,s,V = svd(Y,full_matrices=False)
    #reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    print(U.shape, s.shape, V.shape)
    
    ## here V = V^t, is the output of the svd function
    
    eigvals = s**2
    print('Singular values:', s)
    print('Eigenvalues of the covariance matrix: (variances)', eigvals)
    propV = np.zeros(s.shape)
    totalV = np.sum(eigvals)
    for i in range(s.shape[0]):
        propV[i] = eigvals[i]/totalV
    
    print(V.shape, A.shape)
    
    if corr:
        PCA = np.matmul(V, A_stand)
    else:
        PCA = np.matmul(V, A_centr)
    print(PCA.shape)
    
    #PCA = np.matmul(V, A)
    #print(PCA.shape)
    
    ## Each column of V.T = V, or each row of V=V^t
    for i in range(len(propV)):
        print('--------------')
        print('The principal component with variance proprotion {:.3f}'.format(propV[i])+' is given by the direction:', V[:, i]) 
        print('Its corresponding standard deviation is:', np.std(V[:,i]))
    
    print('--------------')
    print('The expression of the original variables in terms of the principal components is given as follows:')

    print(PCA.T)
    #for i in range(PCA.shape[]):
    #    print(PCA[:, i])
        
    return PCA.T, propV, V


def checkVecs(A):
    corr = np.corrcoef(A)
    pc = np.linalg.eig(corr)
    print('Eigenvectors of correlation matrix:', pc)
    return pc



A=loadData('example.dat')
PCA, lamb, VV = computePCA_corr(A)
PCA2, lamb2, VV2 = computePCA_corr(A, corr = True)

check = checkVecs(A)


