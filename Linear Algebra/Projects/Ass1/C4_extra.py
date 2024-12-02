#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:28:18 2022

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
#import scipy
from scipy.linalg import ldl
from scipy.linalg import solve_triangular

def predictor_substep(n, G, C, g, d, x_0, lambda_0, s_0, corrector = False, mu=0, sigma=0):
    '''
        This function conducts the 1)Predictor substep: 
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
        #print('Condition number of M(KKT)', np.linalg.cond(MK))
        
    
    return dz, F_z0
  
 
############################
## ALTERNATIVE STRATEGIES ##
############################

def strategy1(n, G, C, g, d, x_0, lambda_0, s_0, corrector = False, mu=0, sigma=0):
     m= 2*n
     S = np.zeros((m,m))
     np.fill_diagonal(S, s_0)
     Lam = np.zeros((m,m))
     np.fill_diagonal(Lam, lambda_0)
     
     MK = np.block([
         [G, -C],
         [- C.T, - np.matmul(np.linalg.inv(Lam), S)]
         ])
     
     rl = np.matmul(G, x_0)- g - np.matmul(C,lambda_0)
     rc = s_0 + d - np.matmul(C.T, x_0)
     rs = s_0* lambda_0
     
     F_z0 = np.block([
         [rl],
         [rc],
         [rs]
     ])
     
     F_z1 = np.block([
         [rl],
         [rc - np.matmul(np.linalg.inv(Lam), rs)]
     ])
     
     ## LDL^T factorization:
     lu, d, perm = ldl(MK)
     L_triag = lu[perm, :]
     '''
         This is equivalent to multiplicate the lu by the permutation matrix P
         Note that the matrix lu[perm, :] = P.dot(lu) is the triangular matrix we are looking for 
         where P = I[perm,:]
     '''
     
     ## We solve now two triangular systems and a diagonal direct system
     dy1 = solve_triangular(L_triag, -F_z1[perm], lower=True, unit_diagonal=True)
     diagonal = np.diag(d).reshape(d.shape[0], 1)
     dy2 = np.divide(dy1, diagonal)
     
     dy3 = solve_triangular(L_triag.T, dy2[perm], lower=False, unit_diagonal=True)
     
     ## Now we compute dz1 as the function output
     Lam1 = np.zeros((m,m))
     np.fill_diagonal(Lam1, 1/lambda_0)
     ds = np.matmul(Lam1, -rs- np.matmul(S, dy3[n: (n+m)]))
     
     dz = np.block([
         [dy3[0:n]],
         [dy3[n:(n+m)]],
         [ds]
     ])
     
     if corrector == True:
         dx = dz[0:n]
         dlamb = dz[(n):(n+m)]
         ds = dz[(n + m):(n+m+m)]
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
         F_z1 = np.block([
             [rl],
             [rc - np.matmul(np.linalg.inv(Lam), rs_new)]
         ])
         dy1 = solve_triangular(L_triag, -F_z1[perm], lower=True, unit_diagonal=True)
         diagonal = np.diag(d).reshape(d.shape[0], 1)
         dy2 = np.divide(dy1, diagonal)
         
         dy3 = solve_triangular(L_triag.T, dy2[perm], lower=False, unit_diagonal=True)
         
         ## Now we compute dz1 as the function output
         Lam1 = np.zeros((m,m))
         np.fill_diagonal(Lam1, 1/lambda_0)
         ds = np.matmul(Lam1, -rs_new- np.matmul(S, dy3[n: (n+m)]))
         
         dz = np.block([
             [dy3[0:n]],
             [dy3[n:(n+m)]],
             [ds]
         ])
     
     return dz, F_z0, MK
     

def strategy2(n, G, C, g, d, x_0, lambda_0, s_0, corrector = False, mu=0, sigma=0):
    m= 2*n
    S = np.zeros((m,m))
    np.fill_diagonal(S, s_0)
    Lam = np.zeros((m,m))
    np.fill_diagonal(Lam, lambda_0)
    s_1 = 1/s_0
    S1 = np.zeros((m,m))
    np.fill_diagonal(S1, s_1)
    
    rl = np.matmul(G, x_0)- g - np.matmul(C,lambda_0)
    rc = s_0 + d - np.matmul(C.T, x_0)
    rs = s_0* lambda_0
    
    F_z0 = np.block([
        [rl],
        [rc],
        [rs]
    ])
    
    CS1 = np.matmul(C, S1)
    LamC = np.matmul(Lam, C.T)
    GG = G + np.matmul(CS1, LamC) 
    Lamr2 = np.matmul(Lam, rc)
    rr = - np.matmul(CS1, -rs + Lamr2)
    S1Lam = np.zeros((m,m))
    np.fill_diagonal(S1Lam, s_1*lambda_0)
    
    ## Cholesky factorization
    
    if min(np.linalg.eig(GG)[0]) > 0:
        #print(np.linalg.eig(GG)[0])
        G_chol = np.linalg.cholesky(GG)
        dy1 = solve_triangular(G_chol, -rl- rr, lower=True, unit_diagonal=False)
        dy2 = solve_triangular(G_chol, dy1, lower=False, unit_diagonal=False)
    else:
        dy2 = np.linalg.solve(GG, -rl -rr)
        
        
    Cdx = np.matmul(C.T, dy2)
    
    dlamb = np.matmul(S1, -rs + Lamr2)- np.matmul(S1Lam, Cdx) 
    ds = -rc + Cdx
    
    dz = np.block([
        [dy2],
        [dlamb],
        [ds]
    ])
    
    if corrector == True:
        dx = dz[0:n]
        dlamb = dz[(n):(n+m)]
        ds = dz[(n + m):(n+m+m)]
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
        
        rr = - np.matmul(CS1, -rs_new + Lamr2)
        
        if min(np.linalg.eig(GG)[0]) > 0:
            dy1 = solve_triangular(G_chol, -rl- rr, lower=True, unit_diagonal=False)
            dy2 = solve_triangular(G_chol, dy1, lower=False, unit_diagonal=False)
        else: 
            dy2 = np.linalg.solve(GG, -rl -rr)
            
        Cdx = np.matmul(C.T, dy2)
        
        dlamb = np.matmul(S1, -rs + Lamr2)- np.matmul(S1Lam, Cdx) 
        ds = -rc + Cdx
        
        dz = np.block([
            [dy2],
            [dlamb],
            [ds]
        ])
    
    return dz, F_z0, GG
    
    
 

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



def testProblem(n, method =0, show_prints= False):
    '''
    Here we may define the test problem (like the main function) to test our 
    algorithm.
    
    Parameters
    ----------
    n : dimension of the matrices and vectors we are defining
    method: int that indicates which method may be implemented to solve the corresponding systems
    show_prints: boolean that conditions whether the solution, mse and time computation are printed

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
    x_0 = np.zeros((n, 1))
    s_0 = np.full((m, 1), 1)
    lambda_0 = np.full((m,1), 1)
    z0 = np.block([
        [x_0],
        [lambda_0],
        [s_0]
    ])
    
    
    ########################
    ## STOPPING CRITERION ##
    ########################
    
    epsilon = 1e-16
    maxIterations = 100
    mu_0= 1

    #num_iteration = 0
    
    
    #########################################
    ## ITERATIVELLY APPLYING THE ALGORITHM ##
    #########################################
    
    #while min(abs(rl_0)) > epsilon and min(abs(rc_0)) > epsilon and min(abs(mu_0)) > epsilon and num_iteration < maxIterations:
    for i in range(0, maxIterations):    
        ## 1) First we compute the predictor substep
        if method==0:
            dz0, Fz0 = predictor_substep(n, G, C,g, d, x_0, lambda_0, s_0)
        elif method ==1:
            dz0, Fz0 = strategy1(n, G, C,g, d, x_0, lambda_0, s_0)
        else: 
            dz0, Fz0 = strategy2(n, G, C,g, d, x_0, lambda_0, s_0)
            
            
        dx = dz0[0:n]
        dlamb = dz0[(n+1):(n+m)]
        ds = dz0[(n + m + 1):(n+m+m)]
        
        
        ## 2) Step-size correction subestep:
        alp = Newton_step(lambda_0, dlamb, s_0, ds)
        
        ## 3) Computation of the mu's
        mu, mu2, sigma = compute_mus(s_0, lambda_0, n, dz0, alp)
        
        ## 4) Corrector substep 
        
        if method==0:
            dz1, Fz1 = predictor_substep(n, G, C, g, d,x_0, lambda_0, s_0,corrector = True, mu = mu, sigma = sigma)
        elif method ==1:
            dz1, Fz1, M1 = strategy1(n, G, C, g, d,x_0, lambda_0, s_0, corrector = True, mu = mu, sigma = sigma)
        else: 
            dz1, Fz1, M2 = strategy2(n, G, C, g, d,x_0, lambda_0, s_0, corrector = True, mu = mu, sigma = sigma)
        rl_0 = Fz1[0:n]
        rc_0 = Fz1[(n+1):(n+m)]
        rs_0 = Fz1[(n + m + 1):(n+m+m)]
        
        ## 5) Step-size correction substep
        dx1 = dz1[0:n]
        dlamb1 = dz1[(n+1):(n+m)]
        ds1 = dz1[(n + m + 1):(n+m+m)]
        alp1 = Newton_step(lambda_0, dlamb1, s_0, ds1)
        
        ## 6) Update substep:
        z1 = z0 + 0.95 * alp1 * dz1
        x_0 = z1[0:n]
        lambda_0 = z1[(n):(n+m)]
        s_0 = z1[(n + m ):(n+m+m)]
        ## M_KKT gets updated by introducing the new values 
        # of s, lambda in the corresponding function
        
        if min(abs(rl_0)) < epsilon or min(abs(rc_0)) < epsilon or min(abs(mu)) < epsilon:
            break
    et=time.perf_counter()   
    
    
    mse = ((-g - z1[0:n])**2).mean()
    time_spent = et - st
    
    
    if show_prints == True:
        print('------------ Dimension: ', n)
        print('The solution of the test problem given by the algorithm is: ', z1[0], z1[n+1], z1[n+m+1])
        print('The mean square error between the real and computed solution is:', mse)
        print('The time spent computing this algorithm is:', time_spent)
        print('Iterations performed:', i)
        if method == 1:
            print('Condition number for Strategy 1 matrix:', np.linalg.cond(M1))
        if method == 2:
            print('Condition number for Strategy 2 matrix:', np.linalg.cond(M2))
        
        
    
    return z1, time_spent

    

def compareTimes(n):
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
        n_value3=[]
        for dimension in range(2, n):
            #print('Dimension n=', dimension)
            n_value1.append(dimension)
            z1, time_spent1 = testProblem(dimension)
            computation_time1.append(time_spent1)
            
            z2, time_spent2 = testProblem(dimension, method = 1)
            computation_time2.append(time_spent2)
            
            z3, time_spent3 = testProblem(dimension, method = 2)
            computation_time3.append(time_spent3)
            
        plt.plot(n_value1, computation_time1, 'b', label = 'Standard')
        plt.plot(n_value1, computation_time2, 'g', label = 'Strategy 1')
        plt.plot(n_value1, computation_time3, 'r', label = 'Strategy 2')
        plt.legend(loc="upper left")
        plt.title('Computation Time over dimension n')
        
        
                
compareTimes(100)

#testProblem(4, method = 1, show_prints=True)

