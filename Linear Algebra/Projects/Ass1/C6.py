#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:31:57 2022

@author: flaviaferrusmarimon
"""


'''
    This programe solves the optimization problem for the general case
    following the initial strategy and strategy 1 (with LDLt)
'''

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import ldl
from scipy.linalg import solve_triangular
import scipy.sparse as spsp
from scipy.sparse import csc_matrix, linalg as spspla



def load_matrix(file_name, n, m):
    '''
        This function uploads the matrix in the file file_name with n rows and m columns

    Parameters
    ----------
    file_name : string
    n : number of rows
    m : number of cols

    Returns
    -------
    Matrix

    '''
    
    rows,cols,data=np.loadtxt(file_name,unpack=True)
    rows=rows-1; cols=cols-1;
    A=spsp.coo_matrix((data,(rows,cols)),shape=(n,m))
    cscA = spsp.csc_matrix(A)
    if n==m:
        ## The symmetric matrices are stored just in one triangular side 
        diagonalG = np.zeros((n,n))
        for i in range(0,n):
            diagonalG[i,i] = cscA[i,i]
            
        cscG_lower = cscA.T - diagonalG
        cscG_sym = cscA + cscG_lower
        return cscG_sym 
        
    return A.todense()


def Newton_step(lamb0, dlamb, s0, ds):
    '''
    This function computes alpha so that the step-size
    alpha*dz guarantees feasibility.
    It is used on the step-size correction substeps 
    of the algorithm.
    
    Returns
    -------
    alp : Newton step size, alpha

    '''
    
    alp = 1
    idx_lamb0 = np.array(np.where(dlamb< 0))
    #print(-lamb0[idx_lamb0][0] / dlamb[idx_lamb0][0])
    if idx_lamb0.size >0 :
        alp= min(alp, np.min(-lamb0[idx_lamb0][0] / dlamb[idx_lamb0][0]))
        
    idx_s0 = np.array(np.where(ds<0))
    if idx_s0.size> 0:
        alp= min(alp, np.min(-s0[idx_s0][0] / ds[idx_s0][0]))
     
    return alp

def compute_mus(s0, gamma0, lamb0, n, dz, alp):
    '''
    This function computes the corresponding values
    on step 3 of the algorithm
    
    '''
    p = gamma0.shape[0]
    m=2*n
    dx = dz[0:n]
    dgamm = dz[(n):(n+p)]
    dlamb = dz[(n+p):(n+p+m)]
    ds = dz[(n +p+ m):(n+p+2*m)]
    
    mu = np.dot(s0.T,lamb0) / m
    mu2 = np.dot((s0 + alp*ds).T,(lamb0 + alp*dlamb))/m
    sigma = (mu2/mu)**3
    
    return mu, mu2, sigma


#################################
### FUNCTIONS THAT SOLVE LDLT ###
#################################

def isDiagonal(A, thres= 1e-10):
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if i != j:
                if abs(A[i,j]) < thres:
                    return False
    return True
         
def solveNotDiagonal(A, y, thres = 1e-10):
    blocks = np.zeros((2,2))
    solution = np.zeros((len(y), 1))
    
    i = 0
    while i < A.shape[0]-1:
        if abs(A[i, i+1]) < thres: # and A[i+1, i] < thres:
            solution[i] = y[i]/A[i,i]
            i += 1
        else: 
            ## We have a block here
            blocks[:, :] = A[i:i+2, i:i+2]
            solution[i:i+2] = np.linalg.solve(blocks, y[i:i+2])
            blocks = np.zeros((2,2))
            i += 2
            
    if abs(A[i-1, i]) < thres:
        solution[-1] = (y[-1] / A[-1,-1])
    
    return solution

def algorithm(n, G, C, A, g, d, b, z0, print_condNumber=False, method = 0):
    m = 2*n
    p = A.shape[1]
    thres = 1e-10
    
    x_0= z0[:n]
    gamma_0 = z0[(n):(n+p)]
    lambda_0 =z0[(n+p):(n+p+m)]
    s_0 = z0[(n + p + m):(n+p+2*m)]
    
    S = np.zeros((m,m))
    np.fill_diagonal(S, s_0)
    Lam = np.zeros((m,m))
    np.fill_diagonal(Lam, lambda_0)
    
    ## 1) First we compute the predictor substep
    d = d.reshape(d.shape[0], 1)
    b = b.reshape(b.shape[0], 1)
    rl = np.matmul(G, x_0)+ g -np.matmul(A, gamma_0) - np.matmul(C,lambda_0)
    rA = b - np.matmul(A.T, x_0)
    rc = s_0 + d - np.matmul(C.T, x_0)
    rs = np.zeros((m, 1))
    for i in range(0, m):
        rs[i] = s_0[i]*lambda_0[i]
     
    ## Implement LDLT   
    if method == 1 and min(abs(lambda_0)) > thres:
        #print('Solving LDLt system')
        
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        MK = np.block([
            [G,     -A,     -C],
            [-A.T,  np.zeros((p, p)),   np.zeros((p, m))],
            [-C.T,  np.zeros((m, p)),   -np.matmul(Lam1, S)]
        ])
        if print_condNumber:
            print('Condition number of MKKT for Strategy 1:', np.linalg.cond(MK))
        
        F_z1 = np.block([
            [rl],
            [rA],
            [rc - np.matmul(Lam1, rs)]
        ])
        
        ## LDL^T factorization:
        lu, D, perm = ldl(MK)
        
        L_triag = lu[perm, :]
        '''
            This is equivalent to multiply the lu by the permutation matrix P
            Note that the matrix lu[perm, :] = P.dot(lu) is the triangular matrix we are looking for 
            where P = I[perm,:]
        '''
        dy1 = solve_triangular(L_triag, -F_z1[perm], lower=True, unit_diagonal=True)
        
        if isDiagonal(D) == False:
            #print('Solving not diagonal system')
            
            dy2 = solveNotDiagonal(D, dy1) #.reshape(dy1.shape[0],1)
            dw2 = np.linalg.solve(D, dy1)
            #print('Error committed solving the not diagonal system:', np.linalg.norm(dy2- dw2))
        else: 
            diagonal = np.diag(D).reshape(D.shape[0], 1)
            dy2 = np.divide(dy1, diagonal)
            
        
        dy3 = solve_triangular(L_triag.T, dy2, lower=False, unit_diagonal=True)
        P = np.identity(len(perm))[perm,:]
        dy4 = np.matmul(P.T, dy3) 
        
        ## Now we compute dz
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        ds = np.matmul(Lam1, -rs- np.matmul(S, dy4[(n+p): (n+p+m)]))
        
        #print(ds.shape, dy4.shape)
        dz = np.block([
            [dy4[:n]],
            [dy4[n:(n+p)]],
            [dy4[(n+p):(n+p+m)]],
            [ds]
        ])
            
        
    else: ## method == 0 or not satisfying conditions for LDLt
        MK= np.block([
            [ G,    -A,     -C,     np.zeros((n,m)) ],
            [-A.T,  np.zeros((p,p)),    np.zeros((p,m)),    np.zeros((p,m))],
            [-C.T,  np.zeros((m,p)),    np.zeros((m,m)),    np.identity(m)], 
            [np.zeros((m,n)),   np.zeros((m,p)),     S,     Lam]
        ])
        if print_condNumber:
            print('Condition number of MKKT:', np.linalg.cond(MK))
       
        F_z0 = np.block([
            [rl],
            [rA],
            [rc],
            [rs]
        ])
        dz = np.linalg.solve(MK, -F_z0)
    
    #print('Error comitted solving using LDLt:', np.linalg.norm(dz - dz3))
    
    dx = dz[:n]
    dgamma = dz[(n):(n+p)]
    dlamb = dz[(n+p):(n+p+m)]
    ds = dz[(n +p+ m):(n+p+2*m)]
    
    ## 2) Step-size correction subestep:
    alp = Newton_step(lambda_0, dlamb, s_0, ds)
    
    ## 3) Computation of the mu's
    mu, mu2, sigma = compute_mus(s_0, gamma_0, lambda_0, n, dz, alp)
    
    ## 4) Corrector substep 
    dsdl = np.zeros((m, 1))
    for i in range(0, m):
        dsdl[i] = ds[i]*dlamb[i]
        
    #rs_new = rs + ds*dlamb - mu * sigma
    rs_new = rs + dsdl - mu * sigma
    
    F_z1 = np.block([
        [rl],
        [rA],
        [rc],
        [rs_new]
    ])
    
    if method == 1 and min(abs(lambda_0)) > thres:
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        F_z2 = np.block([
            [rl],
            [rA],
            [rc - np.matmul(Lam1, rs_new)]
        ])
        
        ## We solve now two triangular systems and a diagonal direct system
        dy1 = solve_triangular(L_triag, - F_z2[perm], lower=True, unit_diagonal=True)
        if isDiagonal(D) == False:
            dy2 = solveNotDiagonal(D, dy1) #.reshape(dy1.shape[0],1)
        else: 
            diagonal = np.diag(D).reshape(D.shape[0], 1)
            dy2 = np.divide(dy1, diagonal)
        
        dy3 = solve_triangular(L_triag.T, dy2, lower=False, unit_diagonal=True)
        P = np.identity(len(perm))[perm,:]
        dy4 = np.matmul(P.T, dy3)
            
        ## Now we compute dz
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        ds = np.matmul(Lam1, -rs_new- np.matmul(S, dy4[(n+p): (n+p+m)]))
        
        #print(dy4.shape)
        dz1 = np.block([
            [dy4[0:n]],
            [dy4[n:(n+p)]],
            [dy4[(n+p):(n+p+m)]],
            [ds]
        ])
        
    else: 
        dz1= np.linalg.solve(MK, -F_z1)
    
    ## 5) Step-size correction substep
    dx1 = dz1[0:n]
    dgamm1 = dz1[(n):(n+p)]
    dlamb1 = dz1[(n+p):(n+p+m)]
    ds1 = dz1[(n +p+ m):(n+p+2*m)]
    alp1 = Newton_step(lambda_0, dlamb1, s_0, ds1)
    
    ## 6) Update substep:
    #print(z0.shape, dz1.shape)
    z1 = z0 + 0.95 * alp1 * dz1
    
    
    return z1, F_z1, mu
    
def mse_calculation(x, G, g, problem =1):
    f = (1/2 * np.matmul(np.matmul(x.T, G), x) + np.matmul(g.T, x)).item()
    if problem == 1:
        f_real = 1.15907181*1e4
    else:
        f_real = 1.08751157 * 1e6
        
    return (f- f_real)**2
   


def testProblem(cnumber = False, method = 0, show_prints = False, make_plot = False, problem = 1):
    '''
    Here we may define the test problem (like the main function) to test our 
    algorithm.
    
    Parameters
    ----------
    n : dimension of the matrices and vectors we are defining

    Returns
    -------
    Optimal solution of the test problem.

    '''
    
    ####################
    ## INITIALIZATION ##
    ####################
    
    ## We start the time counter
    st=time.perf_counter()
    
    ## We initialize the test problem:
    ## We have to read the corresponding given matrices
    
    if problem == 1:
        n = 100
        p = 50
        m = 200
        A = load_matrix('./optpr1-20221101/A.dad', n, p)
        C = load_matrix('./optpr1-20221101/C.dad', n, m)
        G = load_matrix('./optpr1-20221101/g.dad', n, n)
        rows_b, b = np.loadtxt('./optpr1-20221101/b.dad', unpack=True)
        rows_d, d = np.loadtxt('./optpr1-20221101/d.dad', unpack=True)

        
    else:
        n = 1000
        p = 500
        m = 2000
        A = load_matrix('./optpr2-20221101/A.dad', n, p)
        C = load_matrix('./optpr2-20221101/C.dad', n, m)
        G = load_matrix('./optpr2-20221101/G.dad', n, n)
        rows_b, b = np.loadtxt('./optpr2-20221101/b.dad', unpack=True)
        rows_d, d = np.loadtxt('./optpr2-20221101/d.dad', unpack=True)
        
    
    g = np.zeros((n, 1))
    x_0 = np.zeros((n, 1))
    s_0 = np.full((m, 1), 1)
    lambda_0 = np.full((m, 1), 1)
    gamm_0 = np.full((p, 1), 1)
    z0 = np.block([
        [x_0],
        [gamm_0],
        [lambda_0],
        [s_0]
    ])
    
    
    if make_plot:
        mse_list =[]
    
    
    ########################
    ## STOPPING CRITERION ##
    ########################
    
    epsilon = 1e-16
    maxIterations = 100
    
    
    #########################################
    ## ITERATIVELLY APPLYING THE ALGORITHM ##
    #########################################
    
    #while min(abs(rl_0)) > epsilon and min(abs(rc_0)) > epsilon and min(abs(mu_0)) > epsilon and num_iteration < maxIterations:
    for i in range(0, maxIterations):   
        
        if method == 0:
            z0, F1, mu =algorithm(n, G, C, A, g, d, b, z0, print_condNumber=cnumber, method = 0)
        else: # method == 1:
            z0, F1, mu =algorithm(n, G, C, A, g, d, b, z0, print_condNumber=cnumber, method = 1)
        
        rl_0 = F1[:n]
        ra_0 = F1[(n):(n+p)]
        rc_0 = F1[(n+p):(n+p+m)]
        rs_0 = F1[(n +p+ m):(n+p+2*m)]
        
        if make_plot:
            mse_list.append(mse_calculation(z0[:n], G, g, problem =problem))
        
        if np.linalg.norm(rl_0) < epsilon or np.linalg.norm(rc_0) < epsilon or np.linalg.norm(ra_0) < epsilon or np.linalg.norm(mu) < epsilon:
            print(np.linalg.norm(rl_0), np.linalg.norm(rc_0), np.linalg.norm(ra_0))
            break
        
        if show_prints:
            print('--- Iteration:', i)
            print( mse_list[-1])
        
    et=time.perf_counter() 
    time_spent = et - st
    if show_prints:
        print('------------ Dimension: ', n)
        print('The mean square error between the real and computed solution is:',  mse_calculation(z0[:n], G, g, problem =problem))
        print('The time spent computing this algorithm is:', time_spent)
        print('Iterations performed:', i)
        
    if make_plot:
        #print(mse_list)
        x_axis = range(len(mse_list))
        plt.plot(x_axis, mse_list)
        
    
    return z0, time_spent


     

## Method 0, initial strategy converges
#testProblem( cnumber = False, method = 0, show_prints = True, make_plot=True, problem = 1)
#testProblem( cnumber = False, method = 0, show_prints = True, make_plot=True, problem = 2)

## Printing condition numbers
#testProblem( cnumber = True, method = 0, show_prints = True, make_plot=True, problem = 1)
#testProblem( cnumber = True, method = 1, show_prints = True, make_plot=True, problem = 1)

## Implemeting Strategy 1: method converges
#testProblem( cnumber = False, method = 1, show_prints = True, make_plot=True, problem = 1)
testProblem( cnumber = False, method = 1, show_prints = True, make_plot=True, problem = 2)

