# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:10:50 2016

@author: lifu
"""
import math
import numpy as np

def xavier_weight_filler(weight):
    """Initialize weight matrix with Xavier method.
    """
    
    r = math.sqrt(6.0 / (weight.shape[0] + weight.shape[1]))
    weight[:] = np.random.uniform(-r, r, weight.shape)
    
def identity_weight_filler(weight):
    """Initialize weight to a identity matrix.
    """
    
    m, n = weight.shape
    if m != n:    
        raise ValueError("weight need to be square matrix.")
    weight[:] = np.eye(m, m)

def gaussian_weight_filler(weight, mu=0, sigma=1.0):
    """Initialize weight matrix using Gaussian distribution.
    """
    
    weight[:] = sigma * np.random.randn(weight.shape) + mu

weight_fillers = {'xavier': xavier_weight_filler,
                  'identity': identity_weight_filler,
                  'gaussian': gaussian_weight_filler}

def constant_bias_filler(bias):
    """Initialize all elements of bias to be zero.
    """
    
    bias[:] = 0
      
bias_fillers = {'constant': constant_bias_filler}
