#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 23:11:37 2022

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
    return cscA

def pageRank(A, n = 36682, m=0.15, tol = 1e-16, iter_max = 1e3):
    indexs = A.indices
    point_ind = A.indptr
    nj = np.zeros((n,1))
    Lj = np.zeros((n,54))
    for j in range(n):
        nj[j] = point_ind[j+1] - point_ind[j]
        for k in range(int(nj[j])):
            Lj[j, k] = indexs[k + point_ind[j]]
    
    
    x0 = np.ones((n, 1))
    
    for k in range(int(iter_max)):
        xk = np.zeros((n, 1))
        
        print('---------------')
        print('Iteration:', k)
        
        for j in range(0,n):
            if(nj[j] == 0):
                xk = xk + x0[j]/n
            else: 
                for i in Lj[j, :int(nj[j])]:
                    i = int(i)
                    xk[i] = xk[i] + x0[j]/nj[j]
        xk = (1-m)*xk + m/n
        
        
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
A = uploadSparseMatrix(matrix_name, n)
xk = pageRank(A)





