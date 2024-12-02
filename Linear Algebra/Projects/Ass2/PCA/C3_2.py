#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:54:52 2022

@author: flaviaferrusmarimon
"""

import pandas as pd
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt


def loadData(name):
    B = pd.read_csv(name)
    B = B.fillna(0)
    
    #print(df_centered.shape)
    B1 = B.to_numpy()
    B2 = B1[:, 1:]
    
    df = pd.DataFrame(B2.T)
    print(df.shape)
    
    print(df.mean())
    print('Centering the data...')
    #center the values in each column of the DataFrame
    df_centered = df.apply(lambda x: x-x.mean())
    
    #view centered DataFrame
    #print(df_centered)
    print(df_centered.mean())
    df_centered = df_centered.T
    
    B3 = np.array( df_centered.to_numpy() ,dtype='float64')
    print(B3.shape)

    array_sample = B.columns[1:]
    
    return B3, array_sample

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
        df= pd.DataFrame(A.T)
        df_stand= df.apply(lambda x: x/x.std())
        df_stand = df_stand.T
        A_stand = np.array(df_stand.to_numpy(), dtype='float64')
        ## Compute the correlation matrix:
        #for i in range(m):
        #    A_stand[i, :] = (A[i,:] - np.repeat(np.mean(A[i, :]), len(A[i, :])) ) /np.std(A[i,:])

        #print(A_stand)
        ## In order to center the data I may use a more computationally affordable method:
            

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
        
        #A_centr = np.zeros(A.shape)
        ## Compute the correlation matrix:
        #for i in range(m):
        #    A_centr[i, :] = (A[i,:] - np.repeat(np.mean(A[i, :]), len(A[i, :])) )
        Y = 1/np.sqrt(n-1)* A.T
    
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
        PCA = np.matmul(V, A)
    print(PCA.shape)
    
    #PCA = np.matmul(V, A)
    #print(PCA.shape)
    
    ## Each column of V.T = V, or each row of V=V^t
    for i in range(len(propV)):
        print('--------------')
        print('The principal component with variance proprotion {:.5f}'.format(propV[i])+' is given by the direction:', V[:, i]) 
        print('Its corresponding standard deviation is:', np.std(V[:,i]))
    
    print('--------------')
    print('The expression of the original variables in terms of the principal components is given as follows:')

    print(PCA.T)
    #for i in range(PCA.shape[]):
    #    print(PCA[:, i])
        
    return PCA.T, propV, V, eigvals


def checkVecs(A):
    corr = np.corrcoef(A)
    pc = np.linalg.eig(corr)
    print('Eigenvectors of correlation matrix:', pc)
    return pc

def storeResults(output_name, PCA, ratioV, V):
    f = open("outputPCA.txt", "w")
    for i in range(PCA.shape[0]):
        print(samples[i], PCA[i, :], ratioV[i])
        f.write(str(samples[i]))
        f.write(',')
        f.write(str(PCA[i, :]))
        f.write(',')
        f.write(str(ratioV[i]))
        f.write('\n')
    f.close()
    
def screePlot(eigvals, ratioV):
    suma =0
    print('------------')
    print('Amount of variance per number of principal components:')
    for i in range(len(ratioV)):
        suma += ratioV[i]
        print(i, suma)
        
    plt.ylabel('Eigenvalues of covariance matrix')
    plt.xlabel('# of Features')
    plt.title('PCA Scree Plot')
    #plt.ylim(0, max(eigvals))
    plt.ylim(0, 4)
    plt.axhline(y = 1, color = 'r', linestyle = '--')
    plt.plot(eigvals)
    plt.show


B, samples =loadData('RCsGoff.csv')

## we want to center in terms of the m variables for each observation 
## (we center each row, or each col of B.T)

PCA, ratioV, V, eigvals = computePCA_corr(B)

print(PCA.shape)
plt.scatter(PCA[:,0], PCA[:,1])
plt.show()
storeResults('output_PCA.txt', PCA, ratioV, V)
screePlot(eigvals)



