#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:51:22 2022

@author: flaviaferrusmarimon
"""

'''
    We may start the  inequality constraints case, i.e. A=0
    In order to test the routines we may consider the following definitions: 
    For this case we consider a particular problem in which we do not 
    have the terms corresponding to the A part, which are gamma, 
    and the 2nd row and col are not included in the first algorithm
'''


import numpy as np
import matplotlib.pyplot as plt


#def predictor_substep(n, G, C, x_0, lambda_0, s_0, rl, rc, rs):
def predictor_substep(n, G, C, g, d, x_0, lambda_0, s_0, corrector = False, mu=0, sigma=0):
    '''
        This function conducts the Predictor substep: 
        it computes the standard Newton step dz=(dx, dlamb, ds)
        by solving the KKT system
    

    Returns
    -------
    dz: array 
        The solution of the KKT system, which accounts for the 
        Newton's step size

    '''
    
    m= 2*n
    S = np.zeros((m,m))
    np.fill_diagonal(S, s_0)
    Lam = np.zeros((m,m))
    np.fill_diagonal(Lam, lambda_0)
    
    MK= np.block([
        [ G, -C, np.zeros((n,m)) ],
        [-C.T , np.zeros((m,m)), np.identity(m)], 
        [np.zeros((m,n)), S, Lam]
    ])
    ## By computing the gradient at the initial point:
    rl = np.matmul(G, x_0)- g - np.matmul(C,lambda_0)
    rc = s_0 + d - np.matmul(C.T, x_0)
    rs = s_0* lambda_0
    
    F_z0 = np.block([
        [rl],
        [rc],
        [rs]
    ])
    dz = np.linalg.solve(MK, -F_z0)
    
    if corrector == True:
        dx = dz[0:n]
        dlamb = dz[(n+1):(n+m)]
        ds = dz[(n + m + 1):(n+m+m)]
        e_vector=np.full((m, 1), 1)
        DS = np.zeros((m,m))
        np.fill_diagonal(DS, ds)
        Dlam = np.zeros((m,m))
        np.fill_diagonal(Dlam, dlamb)
        DsDl_e =np.matmul(np.matmul(DS, Dlam), e_vector)
        '''
            D_s D_lambda e gives a vector that has per coefficients
            the diagonal of D_s*D_lambda which correspond to ds*dlamb
            
            Create the corresponding function that conducts more 
            efficiently this product
                
        '''
        rs_new = - rs - DsDl_e + mu * sigma* e_vector
        F_z0 = np.block([
            [rl],
            [rc],
            [rs_new]
        ])
        dz = np.linalg.solve(MK, -F_z0)
        print('Condition number of M(KKT)', np.linalg.cond(MK))
        
    
    return dz, F_z0
  
  

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



def testProblem(n):
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
    
    ## We initialize the test problem:
    A=np.zeros((n,n))
    m = 2*n
    G = np.identity(n)
    C = np.block([
        [np.identity(n), np.identity(n)]
    ])

    d = np.full((m, 1), -10)
    g = np.random.normal(0,1, size= (n, 1))
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
    #rl_0 = np.full((n, 1), 1)
    #rc_0 = np.full((m, 1), 1)
    #rs_0 = np.full((m, 1), 1)
    #rl_0 = g
    #rc_0 = d
    #rs_0 = np.zeros((m,1))
    mu_0= 1

    #num_iteration = 0
    
    
    #########################################
    ## ITERATIVELLY APPLYING THE ALGORITHM ##
    #########################################
    
    #while min(abs(rl_0)) > epsilon and min(abs(rc_0)) > epsilon and min(abs(mu_0)) > epsilon and num_iteration < maxIterations:
    for i in range(0, maxIterations):    
        ## 1) First we compute the predictor substep
        #dz0 = predictor_substep(n, G, C, x_0, lambda_0, s_0, rl_0, rc_0, rs_0)
        dz0, Fz0 = predictor_substep(n, G, C,g, d, x_0, lambda_0, s_0)
        dx = dz0[0:n]
        dlamb = dz0[(n+1):(n+m)]
        ds = dz0[(n + m + 1):(n+m+m)]
        
        
        ## 2) Step-size correction subestep:
        alp = Newton_step(lambda_0, dlamb, s_0, ds)
        
        ## 3) Computation of the mu's
        mu, mu2, sigma = compute_mus(s_0, lambda_0, n, dz0, alp)
        
        ## 4) Corrector substep 
        #rs_new = - rs_0 - DsDl_e + mu * sigma* e_vector
        dz1, Fz1 = predictor_substep(n, G, C, g, d,x_0, lambda_0, s_0,corrector = True, mu = mu, sigma = sigma)
        rl_0 = Fz1[0:n]
        rc_0 = Fz1[(n+1):(n+m)]
        rs_0 = Fz1[(n + m + 1):(n+m+m)]
        ## 5) Step-size correction substep
        dx1 = dz1[0:n]
        dlamb1 = dz1[(n):(n+m)]
        ds1 = dz1[(n + m ):(n+m+m)]
        alp1 = Newton_step(lambda_0, dlamb1, s_0, ds1)
        
        ## 6) Update substep:
        z1 = z0 + 0.95 * alp1 * dz1
        x_0 = z1[0:n]
        lambda_0 = z1[(n):(n+m)]
        s_0 = z1[(n + m ):(n+m+m)]
        #print(x_0.shape, lambda_0.shape, s_0.shape)
        ## M_KKT gets updated by introducing the new values 
        # of s, lambda in the corresponding function
        
        iteration_mse = ((-g - z1[0:n])**2).mean()
        mse_list.append(iteration_mse)
        
        if min(abs(rl_0)) < epsilon or min(abs(rc_0)) < epsilon or min(abs(mu)) < epsilon:
            break
        
    print('The solution of the test problem given by the algorithm is: ', z1)   
    #mse = ((-g - z1[0:n])**2).mean()
    print('The mean square error between the real and computed solution is:', mse_list[-1])
    print('Iterations performed:', i)
    print('F(z)', rl_0, rc_0)
    
    x_axis = range(len(mse_list))
    plt.plot(x_axis, mse_list)
    
    print('Condition number of G', np.linalg.cond(G))
    
    
    return z1

    
    
testProblem(4)
    






