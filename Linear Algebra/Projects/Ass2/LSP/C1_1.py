#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:15:30 2022

@author: flaviaferrusmarimon
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular, qr
import matplotlib.pyplot as plt
#from scipy import linalg

'''
    This script solves the LSP for the matrices given by the data sets dataset
    and dataset2.csv by computing in each case the corresponding methods. 
    
    First, we may consider the dataset.csv which provides a rank deficient matrix,
    so we may solve the rank deficient LSP by QR factorization and then by using
    the SVD factorization, so we can compare the results
    
'''

def computeQR_RankDeficient(A, b):
    r = np.linalg.matrix_rank(A)
    m = A.shape[0]
    n = A.shape[1]

    ## QR factorization with pivoting:
    Q, R, P = qr(A, pivoting = True)
    
    R1 = R[:r, :r]
    S = R[:r, r:]
    cd = np.matmul(Q.T, b)
    c = cd[:r]
    d = cd[r:]
    
    ## Solving the normal equations to find u_LS
    RR = np.matmul(R1.T, R1)
    alp = np.matmul(R1.T, c)
    chol = np.linalg.cholesky(RR)
    ## remember G is lower triangular so we have to solve two triangular systems:
    y1 = solve_triangular(chol, alp, lower=True, unit_diagonal=False)
    y2 = solve_triangular(chol.T, y1, lower=False, unit_diagonal=False) ## this is u
    vec = np.zeros((n))
    vec[:r]= y2
    x_sol = vec[P]
    
    ## Solving the QR full rank system to find u_LS
    Q_2, R_2, P_2 = qr(R1, pivoting = True)
    yy = np.matmul(Q_2.T, c) 
    u_sol2 = np.linalg.solve(R_2, yy)
    u_sol22 = np.zeros((n))
    u_sol22[:u_sol2.shape[0]] = u_sol2
    x_sol2 = u_sol22[P]
    
    ## Directly slving the linear system to find u_LS:
    u_sol3 = np.linalg.solve(R1, c)
    u_sol32 = np.zeros((n))
    u_sol32[:u_sol3.shape[0]] = u_sol3
    x_sol3 = u_sol32[P]
    
    return x_sol, x_sol2, x_sol3

def computeSVDRankDeficient(A, b, trunc = True, tol = 1e-4):
    r = np.linalg.matrix_rank(A)
    m = A.shape[0]
    n = A.shape[1]
    
    ## Reduced SVD
    u_r, s_r, v_r = np.linalg.svd(A, full_matrices=False)
    
    index = r
    if trunc:
        for i in range(r):
            if s_r[i] > tol:
                index = i
    print(index)
    x_trunc = np.zeros((n))
    ## Truncated SVD decomposition:
    for k in range(r):
        rr = r-k
        s_r1 = np.zeros((rr))
        for i in range(rr):
            s_r1[i] = 1/s_r[i]
        u_r1 = u_r[:, :rr]
        v_r1 = v_r[:n, :rr]
        A1 = np.matmul(v_r1 * s_r1[..., None, :], u_r1.T)
        x_sol = np.matmul(A1, b)
        if rr == index:
            x_trunc = x_sol
        print('r =', r-k)
        print('Solution found:', x_sol)
        print('Norm ||Ax-b||:', np.linalg.norm(b - np.matmul(A, x_sol)))
        print('------------------')
    
    return x_trunc

def computePolyFit(data):
    BB = data.to_numpy()
    N = BB.shape[0]
    B = np.zeros((N, N))
    
    tol = 1e4
    index = N
    value = 0 
    for i in range(N):
        B[i, 0] = 1
        for j in range(1, N):
            B[i,j] = BB[i,0]**j
            if B[i,j] > tol:
                #l = j
                #if i < l:
                #   l=i
                if j < index:
                    index = j
                    value = B[i,j]
                    
    A = np.zeros((index, index))
    A = B[:index, :index]
    y = BB[:index, 1]
    '''
        for i in range(N):
            B[i, 0] = 1
            for j in range(1, N):
                B[i,j] = BB[i,0]**j
        print(B)
    
    p = np.linalg.solve(A, y)
    p1 = np.zeros((index))
    for i in range(index):
        p1[i] = p[index - i-1]
    poli = np.poly1d(p1)
    # Data for plotting
    t = np.arange(0.0, 8.0, 0.01)
    s = poli(t)

    plt.plot(t, s)
    plt.scatter(BB[:, 0], BB[:,1])
    '''
    for i in range(index):
        ind = index - i
        A = np.zeros((ind, ind))
        A = B[:ind, :ind]
        y = BB[:ind, 1]
        p = np.linalg.solve(A, y)
        p1 = np.zeros((ind))
        for i in range(ind):
            p1[i] = p[ind - i-1]
        poli = np.poly1d(p1)
        # Data for plotting
        t = np.arange(0.0, 8.0, 0.01)
        s = poli(t)
    
        plt.plot(t, s, label = ind)
        plt.scatter(BB[:, 0], BB[:,1])
        plt.legend(loc="upper left")

def checkCoeffs( x, y):
    error = np.zeros((x.shape[0]))
    for i in range(x.shape[0]):
        error[i] = np.abs(x[i] - y[i])
    return error

def loadDataSets():
    data1 = pd.read_csv('dades_regressio.csv', header = None)
    AA =data1.to_numpy()
    A = AA[:, :-1]
    b = AA[:,-1]
    
    data2 = pd.read_csv('dades.txt', header = None, sep=' ')
    data2=pd.concat([data2[2],data2[5]], axis = 1)
    
    return A, b, data2

A,b , data = loadDataSets()
x_sol, x_sol2, x_sol3 = computeQR_RankDeficient(A, b)
print(x_sol2)
print('The norm obtained with normal equations is:', np.linalg.norm(b - np.matmul(A, x_sol)))
print('The norm obtained with QR is:', np.linalg.norm(b - np.matmul(A, x_sol2)))
print('The norm obtained directly solving the linear system is:', np.linalg.norm(b - np.matmul(A, x_sol3)))
#print('Checking coefficient by coefficient error commited:', checkCoeffs(b, np.matmul(A, x_sol2)))
   

x_trunc = computeSVDRankDeficient(A, b)
print(x_trunc)
    
computePolyFit(data)
    
    
    
    
