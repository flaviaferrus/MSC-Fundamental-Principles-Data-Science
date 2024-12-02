#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 17:08:12 2022

@author: flaviaferrusmarimon
"""


'''
    This programme solves the optimization problem for the general case
'''

import numpy as np
import time
import matplotlib.pyplot as plt
#import scipy
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
    #print(lamb0.shape)
    #print(lamb0[idx_lamb0].shape)
    #print(lamb0[idx_lamb0][0].shape)
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
    g = g.reshape(g.shape[0], 1)
    #   print(g.shape)
    b = b.reshape(b.shape[0], 1)
    rl = np.matmul(G, x_0)+ g -np.matmul(A, gamma_0) - np.matmul(C,lambda_0)
    rA = b - np.matmul(A.T, x_0)
    rc = s_0 + d - np.matmul(C.T, x_0)
    #print(s_0.shape, lambda_0.shape)
    #rs = s_0 * lambda_0
    rs = np.zeros((m, 1))
    for i in range(0, m):
        rs[i] = s_0[i]*lambda_0[i]
    
    ### Different strategies applied
    MK= np.block([
        [ G,    -A,     -C,     np.zeros((n,m)) ],
        [-A.T,  np.zeros((p,p)),    np.zeros((p,m)),    np.zeros((p,m))],
        [-C.T,  np.zeros((m,p)),    np.zeros((m,m)),    np.identity(m)], 
        [np.zeros((m,n)),   np.zeros((m,p)),     S,     Lam]
    ])
    if print_condNumber:
        print('Condition number of MKKT:', np.linalg.cond(MK))
    
    #print(rl.shape, rA.shape, rc.shape, rs.shape)
    F_z0 = np.block([
        [rl],
        [rA],
        [rc],
        [rs]
    ])
    
        
    if method == 1 and min(abs(lambda_0)) > thres:
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        MK = np.block([
            [G,     -A,     -C],
            [-A.T,  np.zeros((p, p)),   np.zeros((p, m))],
            [-C.T,  np.zeros((m, p)),   -np.matmul(Lam1, S)]
        ])
        if print_condNumber:
            print('Condition number of MKKT:', np.linalg.cond(MK))
        
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
        ## We solve now two triangular systems and a diagonal direct system
        dy1 = solve_triangular(L_triag, -F_z1[perm], lower=True, unit_diagonal=True)
        diagonal = np.diag(D).reshape(D.shape[0], 1)
        dy2 = np.divide(dy1, diagonal)
        dy3 = solve_triangular(L_triag.T, dy2, lower=False, unit_diagonal=True)
        dy4 =  dy3[perm]
        
        ## Now we compute dz
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        ds = np.matmul(Lam1, -rs- np.matmul(S, dy4[(n+m): (n+2*m)]))
    
        dz = np.block([
            [dy4[0:n]],
            [dy4[n:(n+p)]],
            [dy4[(n+p):(n+p+m)]],
            [ds]
        ])
        
        
        
    elif method == 2 and min(abs(lambda_0)) > thres:
        s_1 = 1/s_0
        S1 = np.zeros((m,m))
        np.fill_diagonal(S1, s_1)
        
        S1Lam = np.zeros((m,m))
        np.fill_diagonal(S1Lam, s_1*lambda_0)
        GG = G + np.matmul(np.matmul(C, S1Lam), C.T) 
        rr = - np.matmul(np.matmul(C, S1), -rs + np.matmul(Lam, rc))
        
        MK2 = np.block([
            [GG,     -A],
            [-A.T,  np.zeros((p, p))]
        ])
        
        if min(np.linalg.eig(MK2)) > thres:
            M_chol = np.linalg.cholesky(MK2)
            dy1 = solve_triangular(M_chol, -rl- rr, lower=True, unit_diagonal=False)
            dy2 = solve_triangular(M_chol.T, dy1, lower=False, unit_diagonal=False)
            
        else:
            dy2 = np.linalg.solve(GG, -rl -rr)
            
        Cdx = np.matmul(C.T, dy2[:n])
        dlamb = np.matmul(S1, -rs + np.matmul(Lam, rc))- np.matmul(S1Lam, Cdx) 
        ds = -rc + Cdx
        
        dgamm = np.linalg.solve(-A.T, -rA)
        dz = np.block([
            [dy2],
            [dgamm],
            [dlamb],
            [ds]
        ])
        
        
    else:
        dz = np.linalg.solve(MK, -F_z0)
    
    dx = dz[:n]
    dgamma = dz[(n):(n+p)]
    dlamb = dz[(n+p):(n+p+m)]
    ds = dz[(n +p+ m):(n+p+2*m)]
    
    ## 2) Step-size correction subestep:
    alp = Newton_step(lambda_0, dlamb, s_0, ds)
    
    ## 3) Computation of the mu's
    mu, mu2, sigma = compute_mus(s_0, gamma_0, lambda_0, n, dz, alp)
    
    ## 4) Corrector substep 
    #print(rs.shape, ds.shape, dlamb.shape)
    
    dsdl = np.zeros((m, 1))
    for i in range(0, m):
        dsdl[i] = ds[i]*dlamb[i]
        
    #print(dsdl.shape)
    
    #print(ds*dlamb)
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
        diagonal = np.diag(D).reshape(D.shape[0], 1)
        dy2 = np.divide(dy1, diagonal)
        dy3 = solve_triangular(L_triag.T, dy2, lower=False, unit_diagonal=True)
        dy4 = dy3[perm]
        
        ## Now we compute dz
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        ds = np.matmul(Lam1, -rs_new- np.matmul(S, dy4[n: (n+m)]))
        
        dz1 = np.block([
            [dy4[0:n]],
            [dy4[n:(n+p)]],
            [dy4[(n+p):(n+p+m)]]
            [ds]
        ])
        
    elif method == 2 and min(abs(lambda_0)) > thres:
        rr = - np.matmul(np.matmul(C, S1), -rs_new + np.matmul(Lam, rc))
        
        if min(np.linalg.eig(MK2)) > thres:
            dy1 = solve_triangular(M_chol, -rl- rr, lower=True, unit_diagonal=False)
            dy2 = solve_triangular(M_chol.T, dy1, lower=False, unit_diagonal=False)
        else:
            dy2 = np.linalg.solve(GG, -rl -rr)
        Cdx = np.matmul(C.T, dy2[:n])
        dlamb = np.matmul(S1, -rs_new + np.matmul(Lam, rc))- np.matmul(S1Lam, Cdx) 
        ds = -rc + Cdx
        dgamm = np.linalg.solve(-A.T, -rA)
        
        dz = np.block([
            [dy2],
            [dgamm],
            [dlamb],
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
   


def testProblem( cnumber = False, method = 0, show_prints = False, make_plot = False, problem = 1):
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
        G = load_matrix('./optpr1-20221101/G.dad', n, n)
        rows_b, b = np.loadtxt('./optpr1-20221101/b.dad', unpack=True)
        rows_d, d = np.loadtxt('./optpr1-20221101/d.dad', unpack=True)
        #rows_g, g = np.loadtxt('./optpr1-20221101/g_minus.dad', unpack=True)

        
    else:
        n = 1000
        p = 500
        m = 2000
        A = load_matrix('./optpr2-20221101/A.dad', n, p)
        C = load_matrix('./optpr2-20221101/C.dad', n, m)
        G = load_matrix('./optpr2-20221101/G.dad', n, n)
        rows_b, b = np.loadtxt('./optpr2-20221101/b.dad', unpack=True)
        rows_d, d = np.loadtxt('./optpr2-20221101/d.dad', unpack=True)
        #rows_g, g = np.loadtxt('./optpr1-20221101/g_minus.dad', unpack=True)
        
    
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
        elif method == 1:
            z0, F1, mu =algorithm(n, G, C, A, g, d, b, z0, print_condNumber=cnumber, method = 1)
        else: ## method== 2
            z0, F1, mu =algorithm(n, G, C, A, g, d, b, z0, print_condNumber=cnumber, method = 2)
        
        rl_0 = F1[:n]
        ra_0 = F1[(n):(n+p)]
        rc_0 = F1[(n+p):(n+p+m)]
        rs_0 = F1[(n +p+ m):(n+p+2*m)]
        
        if make_plot:
            mse_list.append(mse_calculation(z0[:n], G, g, problem =problem))
        
        #if min(abs(rl_0)) < epsilon or min(abs(rc_0)) < epsilon or min(abs(mu)) < epsilon:
        if np.linalg.norm(rl_0) < epsilon or np.linalg.norm(rc_0) < epsilon or np.linalg.norm(ra_0) < epsilon or np.linalg.norm(mu) < epsilon:
            print(np.linalg.norm(rl_0), np.linalg.norm(rc_0), np.linalg.norm(ra_0))
            break
        
        if show_prints:
            print('------------------ \n ------ Iteration:', i)
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


def compareTimes(n, cnumber = False, show_prints=False, problem = 1):
    '''
    This function solves the test problem for different values of n
    

    Parameters
    ----------
    n : int until which we want to compute the computation time (starting by n= 2)

    Returns
    -------
    None.
    Computation time plot

    '''
    
    print('------------ Problem: ', problem)
    
    if n < 2:
        print('You need to introduce a value greater than 2, try again. \n')
        return 0
    
    else:
        computation_time1 =[]
        n_value1=[]
        computation_time2 =[]
        computation_time3 =[]
        n_value3=[]
        for dimension in range(2, n):
            #print('Dimension n=', dimension)
            n_value1.append(dimension)
            z1, time_spent1 = testProblem(dimension, method = 0, show_prints=show_prints, make_plot=False, problem = problem)
            computation_time1.append(time_spent1)
            
            z2, time_spent2 = testProblem(dimension, method = 1, show_prints=show_prints, make_plot=False, problem = problem)
            computation_time2.append(time_spent2)
            
            z3, time_spent3 = testProblem(dimension, method = 2, show_prints=show_prints, make_plot=False, problem = problem)
            computation_time3.append(time_spent3)
            
        plt.plot(n_value1, computation_time1, 'b', label = 'Standard')
        plt.plot(n_value1, computation_time2, 'g', label = 'Strategy 1')
        plt.plot(n_value1, computation_time3, 'r', label = 'Strategy 2')
        plt.legend(loc="upper left")
        plt.title('Computation Time over dimension n')
        


## Method 0, initial strategy converges
#testProblem(cnumber = False, method = 0, show_prints = True, make_plot=True, problem = 1)
testProblem(cnumber = False, method = 0, show_prints = True, make_plot=True, problem = 2)



 