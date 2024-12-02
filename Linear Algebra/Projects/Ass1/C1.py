#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:42:30 2022

@author: flaviaferrusmarimon
"""

import numpy as np

def Newton_step(lamb0, dlamb, s0, ds):
    '''
    This function is used to compute \alpha for the step-size
    correction substeps for the Newton step-size substep algorithm

    Parameters
    ----------
    lamb0 : inequality constraints initial conditions
    dlamb : lambda step
    s0 : C^T x - d for x_0 initial point
    ds : s step

    Returns
    -------
    alp : TYPE
        DESCRIPTION.

    '''
    alp = 1
    idx_lamb0 = np.array(np.where(dlamb< 0))
    if idx_lamb0.size >0 :
        alp= min(alp, np.min(-lamb0[idx_lamb0] / dlamb[idx_lamb0]))
        
    idx_s0 = np.array(np.where(ds<0))
    if idx_s0.size> 0:
        alp= min(alp, np.min(-s0[idx_s0] / ds[idx_s0]))
     
    return alp