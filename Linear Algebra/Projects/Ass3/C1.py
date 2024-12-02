#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 20:04:47 2022

@author: flaviaferrusmarimon
"""

import numpy as np
import scipy.sparse as spsp
from scipy.sparse import csc_matrix, linalg as spspla

def uploadSparseMatrix(matrix_name, n):
    rows,cols=np.loadtxt(matrix_name, unpack = True) #,skiprows=83,unpack=True)
    rows=rows-1; cols=cols-1
    data = np.repeat(1, len(rows))
    A=spsp.coo_matrix((data,(rows,cols)),shape=(n,n))
    cscA = spsp.csc_matrix(A)
    return cscA, rows, cols
    
def multSparse(rows, cols, x0):
    y = np.zeros((x0.shape))
    ## We seek to compute A*x
    for i in range(len(rows)):
        row = int(rows[i])
        col = int(cols[i])
        #print(row, col)
        y[row] +=  x0[col] 
        ## A[rows[i], cols[i]]= 1 always, then this is equivalent to y[row] + A[row, col] * x0[col]
    #print(y)
    return y

def pageRank(A, rows, cols, n = 36682, m=0.15, tol = 1e-16, iter_max = 1e3):
    z = np.zeros((n,1))
    d = np.zeros((n,1))
    n_j = np.zeros((n,1))
    
    for j in range(n):
        n_j[j] = np.sum(A[:, j])
        if n_j[j] > 0:
            d[j] = 1/n_j[j]
            z[j] = m/n
        else:
            d[j] = 0
            z[j] = 1/n
    
    
    x0 = np.ones((n, 1))
    
    for k in range(int(iter_max)):
        xk = np.zeros((n, 1))
        
        print('---------------')
        print('Iteration:', k)
        #print('x0:', x0)
        zx = np.matmul(z.T, x0)
        ezx = np.repeat(zx, x0.shape[0])
        dx = (d*x0)
         
        GDx = multSparse(rows,cols, dx)
        
        for i in range(GDx.shape[0]):
            xk[i] = (1-m) * GDx[i] + zx[0]
        #print(x0)
        print('xk:', xk)
        print(np.linalg.norm(xk - x0, ord = np.inf))
            
        if np.linalg.norm(xk - x0, ord = np.inf) < tol:
            break
        else: 
            x0 = xk  
            #print('New x0:', x0)
    return xk



matrix_name = "./p2p-Gnutella30/p2p-Gnutella30.mtx"
n = 36682
m = 0.15
A, rows, cols = uploadSparseMatrix(matrix_name, n)
xk = pageRank(A, rows, cols)




