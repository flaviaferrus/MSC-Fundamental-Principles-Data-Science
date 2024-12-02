#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:42:20 2022

@author: flaviaferrusmarimon
"""

'''
    This is a modification of program C2 in which the time of the computation
    of the solution is also computed
'''

import numpy as np
import time
import matplotlib.pyplot as plt


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

def algorithm(n, G, C,g, d, z0, print_condNumber=False):
    m = 2*n
    '''
    z0 = np.block([
        [x_0],
        [lambda_0],
        [s_0]
    ])
    '''
    x_0= z0[:n]
    lambda_0 = z0[(n):(n+m)]
    s_0 = z0[(n + m):(n+m+m)]
    S = np.zeros((m,m))
    np.fill_diagonal(S, s_0)
    Lam = np.zeros((m,m))
    np.fill_diagonal(Lam, lambda_0)
    
    ## 1) First we compute the predictor substep
    MK= np.block([
        [ G, -C, np.zeros((n,m)) ],
        [-C.T , np.zeros((m,m)), np.identity(m)], 
        [np.zeros((m,n)), S, Lam]
    ])
    if print_condNumber:
        print('Condition number of MKKT:', np.linalg.cond(MK))
    
    d = d.reshape(d.shape[0], 1)
    rl = np.matmul(G, x_0)+ g - np.matmul(C,lambda_0)
    rc = s_0 + d - np.matmul(C.T, x_0)
    rs = s_0* lambda_0
    
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
    rs_new = rs + ds*dlamb - mu * sigma
    F_z1 = np.block([
        [rl],
        [rc],
        [rs_new]
    ])
    dz1= np.linalg.solve(MK, -F_z1)
    
    ## 5) Step-size correction substep
    dx1 = dz1[0:n]
    dlamb1 = dz1[(n):(n+m)]
    ds1 = dz1[(n + m):(n+m+m)]
    alp1 = Newton_step(lambda_0, dlamb1, s_0, ds1)
    
    ## 6) Update substep:
    z1 = z0 + 0.95 * alp1 * dz1
    
    
    return z1, F_z1, mu
    


def testProblem(n, cnumber = False):
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
        
        z0, F1, mu =algorithm(n, G, C,g, d, z0, print_condNumber=cnumber)
        
        #z0 = z1
        
        rl_0 = F1[:n]
        rc_0 = F1[(n):(n+m)]
        rs_0 = F1[(n + m):(n+m+m)]
        #iteration_mse = ((-g - z1[0:n])**2).mean()
        #iteration_mse = np.linalg.norm(g+z0[:n])
        #mse_list.append(iteration_mse)
        
        if min(abs(rl_0)) < epsilon or min(abs(rc_0)) < epsilon or min(abs(mu)) < epsilon:
        #if np.linalg.norm(rl_0) < epsilon or np.linalg.norm(rc_0) < epsilon or np.linalg.norm(mu) < epsilon:
            break
        
    et=time.perf_counter() 
    print('------------ Dimension: ', n)
    time_spent = et - st
    #mse = ((g + z1[0:n])**2).mean()
    print('The mean square error between the real and computed solution is:',  np.linalg.norm(g+z0[:n]))
    print('The time spent computing this algorithm is:', time_spent)
    print('Iterations performed:', i)
    #print('The solution of the test problem given by the algorithm is: ', z0)   
    #mse = ((-g - z1[0:n])**2).mean()
    
    #x_axis = range(len(mse_list))
    #plt.plot(x_axis, mse_list)
    
    
    return z0, time_spent


def compareTimes(n, cnumber = False):
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
        computation_time =[]
        n_value=[]
        for dimension in range(3, n):
            n_value.append(dimension)
            z, time_spent = testProblem(dimension, cnumber = cnumber)
            computation_time.append(time_spent)
            
        plt.plot(n_value, computation_time)
        plt.title('Time computation over dimension n')
        

#compareTimes(100, cnumber=True)
compareTimes(100)



