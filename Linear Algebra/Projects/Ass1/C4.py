#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:39:01 2022

@author: flaviaferrusmarimon
"""


'''
    This is a modification of program C3 in which the time of the computation
    of the solution is also computed, and the algorithm is conducted using two
    alternative strategies to sole the corresponding KKT system. 
    Therefore, we are modifying substeps 1 and 4 from the algorithm 
'''


import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import ldl
from scipy.linalg import solve_triangular
from scipy.linalg import cho_factor, cho_solve


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
    if idx_lamb0.size >0 :
        alp= min(alp, np.min(-lamb0[idx_lamb0] / dlamb[idx_lamb0]))
        
    idx_s0 = np.array(np.where(ds<0))
    if idx_s0.size> 0:
        alp= min(alp, np.min(-s0[idx_s0] / ds[idx_s0]))
     
    return alp

def compute_mus(s0, lamb0, n, dz, alp):
    '''
    This function computes the corresponding values
    on step 3 of the algorithm
    
    '''
    
    m=2*n
    dx = dz[0:n]
    dlamb = dz[(n):(n+m)]
    ds = dz[(n + m):(n+m+m)]
    
    mu = np.dot(s0.T,lamb0) / m
    mu2 = np.dot((s0 + alp*ds).T,(lamb0 + alp*dlamb))/m
    sigma = (mu2/mu)**3
    
    return mu, mu2, sigma

def algorithm(n, G, C, g, d, z0, print_condNumber=False, method = 0):
    m = 2*n
    thres = 1e-10
    
    x_0= z0[:n]
    lambda_0 = z0[(n):(n+m)]
    s_0 = z0[(n + m):(n+m+m)]
    
    S = np.zeros((m,m))
    np.fill_diagonal(S, s_0)
    Lam = np.zeros((m,m))
    np.fill_diagonal(Lam, lambda_0)
    
    ## 1) First we compute the predictor substep
    d = d.reshape(d.shape[0], 1)
    rl = np.matmul(G, x_0)+ g - np.matmul(C,lambda_0)
    rc = s_0 + d - np.matmul(C.T, x_0)
    rs = s_0* lambda_0
    
    ### Different strategies applied
    
    ## Implement LDLT:
    if method == 1 and min(abs(lambda_0)) > thres:
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        MK = np.block([
            [G, -C],
            [- C.T, - np.matmul(Lam1, S)]
        ])
        if print_condNumber:
            print('Condition number of MKKT: for LDLt', np.linalg.cond(MK))
        F_z1 = np.block([
            [rl],
            [rc - np.matmul(Lam1, rs)]
        ])
        
        ## LDL^T factorization:
        lu, d, perm = ldl(MK)
        
        '''
            This is equivalent to multiply the lu by the permutation matrix P
            Note that the matrix lu[perm, :] = P.dot(lu) is the triangular matrix we are looking for 
            where P = I[perm,:]
        '''
        
        L_triag = lu[perm, :]
        
        ## We solve now two triangular systems and a diagonal direct system
        dy1 = solve_triangular(L_triag, -F_z1[perm], lower=True, unit_diagonal=True)
        diagonal = np.diag(d).reshape(d.shape[0], 1)
        dy2 = np.divide(dy1, diagonal)
        dy3 = solve_triangular(L_triag.T, dy2, lower=False, unit_diagonal=True)
        dy4 = dy3[perm]
        
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        ds = np.matmul(Lam1, -rs- np.matmul(S, dy4[n: (n+m)]))
    
        dz = np.block([
            [dy4[0:n]],
            [dy4[n:(n+m)]],
            [ds]
        ])
        
    #else: #strategy 2
    elif method == 2 and min(abs(lambda_0)) > thres:
        s_1 = 1/s_0
        S1 = np.zeros((m,m))
        np.fill_diagonal(S1, s_1)
        
        '''
            CS1 = np.matmul(C, S1)
            LamC = np.matmul(Lam, C.T)
            Lamr2 = np.matmul(Lam, rc)
        '''
        S1Lam = np.zeros((m,m))
        np.fill_diagonal(S1Lam, s_1*lambda_0)
        GG = G + np.matmul(np.matmul(C, S1Lam), C.T) 
        rr = - np.matmul(np.matmul(C, S1), -rs + np.matmul(Lam, rc))
        
        ## Cholesky factorization
        #print(np.linalg.eig(GG)[0])
        if min(np.linalg.eig(GG)[0]) > thres:
            G_chol = np.linalg.cholesky(GG)
            dy1 = solve_triangular(G_chol, -rl- rr, lower=True, unit_diagonal=False)
            dy2 = solve_triangular(G_chol.T, dy1, lower=False, unit_diagonal=False)
        else:
            dy2 = np.linalg.solve(GG, -rl -rr)
        Cdx = np.matmul(C.T, dy2)
        dlamb = np.matmul(S1, -rs + np.matmul(Lam, rc))- np.matmul(S1Lam, Cdx) 
        ds = -rc + Cdx
        
        dz = np.block([
            [dy2],
            [dlamb],
            [ds]
        ])
    
    elif method == 3 and min(abs(lambda_0)) > thres:
        s_1 = 1/s_0
        S1 = np.zeros((m,m))
        np.fill_diagonal(S1, s_1)
        
        S1Lam = np.zeros((m,m))
        np.fill_diagonal(S1Lam, s_1*lambda_0)
        GG = G + np.matmul(np.matmul(C, S1Lam), C.T) 
        rr = - np.matmul(np.matmul(C, S1), -rs + np.matmul(Lam, rc))
        
        ## Cholesky factorization
        #print(np.linalg.eig(GG)[0])
        if min(np.linalg.eig(GG)[0]) > thres:
            G_ch, low = cho_factor(GG)
            dy2 = cho_solve((G_ch, low), -rl-rr)
        else:
            dy2 = np.linalg.solve(GG, -rl -rr)
        Cdx = np.matmul(C.T, dy2)
        dlamb = np.matmul(S1, -rs + np.matmul(Lam, rc))- np.matmul(S1Lam, Cdx) 
        ds = -rc + Cdx
        
        dz = np.block([
            [dy2],
            [dlamb],
            [ds]
        ])
    
    else: 
        MK= np.block([
            [ G, -C, np.zeros((n,m)) ],
            [-C.T , np.zeros((m,m)), np.identity(m)], 
            [np.zeros((m,n)), S, Lam]
        ])
        if print_condNumber:
            print('Condition number of MKKT:', np.linalg.cond(MK))
        
        F_z0 = np.block([
            [rl],
            [rc],
            [rs]
        ])
        dz = np.linalg.solve(MK, -F_z0)
    
    dx = dz[:n]
    dlamb = dz[(n):(n+m)]
    ds = dz[(n + m):(n+m+m)]
    
    ## 2) Step-size correction subestep:
    alp = Newton_step(lambda_0, dlamb, s_0, ds)
    
    ## 3) Computation of the mu's
    mu, mu2, sigma = compute_mus(s_0, lambda_0, n, dz, alp)
    
    
    ## 4) Corrector substep 
    dsdl = np.zeros((m, 1))
    for i in range(0, m):
        dsdl[i] = ds[i]*dlamb[i]
        
    #rs_new = rs + ds*dlamb - mu * sigma
    rs_new = rs + dsdl - mu * sigma
    F_z1 = np.block([
        [rl],
        [rc],
        [rs_new]
    ])
    
      
    if method == 1 and abs(min(lambda_0)) > thres:
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        F_z2 = np.block([
            [rl],
            [rc - np.matmul(Lam1, rs_new)]
        ])
        ## We solve now two triangular systems and a diagonal direct system
        dy1 = solve_triangular(L_triag, - F_z2[perm], lower=True, unit_diagonal=True)
        diagonal = np.diag(d).reshape(d.shape[0], 1)
        dy2 = np.divide(dy1, diagonal)
        dy3 = solve_triangular(L_triag.T, dy2, lower=False, unit_diagonal=True)
        dy4 =  dy3[perm]
        
        ## Now we compute dz
        Lam1 = np.zeros((m,m))
        np.fill_diagonal(Lam1, 1/lambda_0)
        ds = np.matmul(Lam1, -rs_new- np.matmul(S, dy4[n: (n+m)]))
        
        dz1 = np.block([
            [dy4[0:n]],
            [dy4[n:(n+m)]],
            [ds]
        ])
        
    elif method == 2 and min(abs(lambda_0)) > thres:
        rr = - np.matmul(np.matmul(C, S1), -rs_new + np.matmul(Lam, rc))
        if min(np.linalg.eig(GG)[0]) > 0:
            dy1 = solve_triangular(G_chol, -rl- rr, lower=True, unit_diagonal=False)
            dy2 = solve_triangular(G_chol.T, dy1, lower=False, unit_diagonal=False)
        else:
            dy2 = np.linalg.solve(GG, -rl -rr)
        Cdx = np.matmul(C.T, dy2)
        dlamb = np.matmul(S1, -rs_new + np.matmul(Lam, rc))- np.matmul(S1Lam, Cdx) 
        ds = -rc + Cdx
        
        dz1 = np.block([
            [dy2],
            [dlamb],
            [ds]
        ])
        
    elif method == 3 and min(abs(lambda_0)) > thres:
        rr = - np.matmul(np.matmul(C, S1), -rs_new + np.matmul(Lam, rc))
        if min(np.linalg.eig(GG)[0]) > 0:
            G_ch, low = cho_factor(GG)
            dy2 = cho_solve((G_ch, low), -rl-rr)
        else:
            dy2 = np.linalg.solve(GG, -rl -rr)
        Cdx = np.matmul(C.T, dy2)
        dlamb = np.matmul(S1, -rs_new + np.matmul(Lam, rc))- np.matmul(S1Lam, Cdx) 
        ds = -rc + Cdx
        
        dz1 = np.block([
            [dy2],
            [dlamb],
            [ds]
        ])
        
    
        
    else:
        dz1= np.linalg.solve(MK, -F_z1)
          
    
    ## 5) Step-size correction substep
    dx1 = dz1[0:n]
    dlamb1 = dz1[(n):(n+m)]
    ds1 = dz1[(n + m):(n+m+m)]
    alp1 = Newton_step(lambda_0, dlamb1, s_0, ds1)
    
    ## 6) Update substep:
    z1 = z0 + 0.95 * alp1 * dz1
    
    return z1, F_z1, mu
    


def testProblem(n, cnumber = False, method = 0, show_prints = False, make_plot = False):
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
    A=np.zeros((n,n))
    m = 2*n
    G = np.identity(n)
    C = np.block([
        [np.identity(n), np.identity(n)]
    ])

    d = np.full((m, 1), -10)
    g = np.random.normal(0,1, size= (n, 1))
    #print(g)
    x_0 = np.zeros((n, 1))
    s_0 = np.full((m, 1), 1)
    lambda_0 = np.full((m,1), 1)
    z0 = np.block([
        [x_0],
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
            z0, F1, mu =algorithm(n, G, C, g, d, z0, print_condNumber=cnumber, method = 0)
        elif method == 1:
            z0, F1, mu =algorithm(n, G, C, g, d, z0, print_condNumber=cnumber, method = 1)
        else: ## method== 2
            z0, F1, mu =algorithm(n, G, C, g, d, z0, print_condNumber=cnumber, method = 2)
        
        rl_0 = F1[:n]
        rc_0 = F1[(n):(n+m)]
        rs_0 = F1[(n + m):(n+m+m)]
        
        if make_plot:
            mse_list.append(np.linalg.norm(g+z0[:n]))
        
        if min(abs(rl_0)) < epsilon or min(abs(rc_0)) < epsilon or min(abs(mu)) < epsilon:
            break
        
    et=time.perf_counter() 
    time_spent = et - st
    if show_prints:
        print('------------ Dimension: ', n)
        print('The mean square error between the real and computed solution is:',  np.linalg.norm(g+z0[:n]))
        print('The time spent computing this algorithm is:', time_spent)
        print('Iterations performed:', i)
        
    if make_plot:
        x_axis = range(len(mse_list))
        plt.plot(x_axis, mse_list)
        
    
    return z0, time_spent


def compareTimes(n, cnumber = False, show_prints=False):
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
    
    if n < 2:
        print('You need to introduce a value greater than 2, try again. \n')
        return 0
    
    else:
        computation_time1 =[]
        n_value1=[]
        computation_time2 =[]
        computation_time3 =[]
        # computation_time4 =[]
        
        for dimension in range(2, n):
            #print('Dimension n=', dimension)
            n_value1.append(dimension)
            z1, time_spent1 = testProblem(dimension, method = 0, show_prints=show_prints)
            computation_time1.append(time_spent1)
            
            z2, time_spent2 = testProblem(dimension, method = 1, show_prints=show_prints)
            computation_time2.append(time_spent2)
            
            z3, time_spent3 = testProblem(dimension, method = 2, show_prints=show_prints)
            computation_time3.append(time_spent3)
            
            # z4, time_spent4 = testProblem(dimension, method = 3, show_prints=show_prints)
            # computation_time4.append(time_spent4)
            
        plt.plot(n_value1, computation_time1, 'b', label = 'Standard')
        plt.plot(n_value1, computation_time2, 'g', label = 'Strategy 1')
        plt.plot(n_value1, computation_time3, 'r', label = 'Strategy 2')
        #plt.plot(n_value1, computation_time4, 'p', label = 'Strategy 2.2')
        plt.legend(loc="upper left")
        plt.title('Computation Time over dimension n')
        

#compareTimes(100)
compareTimes(100, cnumber = False, show_prints = True)
#testProblem(10, cnumber = False, method = 0, show_prints = True, make_plot=True)
#testProblem(10, cnumber = False, method = 1, show_prints = True, make_plot=True)
#testProblem(10, cnumber = False, method = 2, show_prints = True, make_plot=True)
 