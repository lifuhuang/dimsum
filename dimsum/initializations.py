# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:10:50 2016

@author: lifu
"""
import math

import numpy as np

def get(name):
    """Return a array filler according to name.
    """
    
    return _array_fillers.get(name, None)

def _xavier_filler(weight):
    """Initialize array with Xavier method.
    """
    
    r = math.sqrt(6.0 / (weight.shape[0] + weight.shape[1]))
    weight[:] = np.random.uniform(-r, r, weight.shape)
    
def _identity_filler(weight):
    """Initialize array to a identity matrix.
    
    Only applies to square matrix.
    """
    
    m, n = weight.shape
    if m != n:    
        raise ValueError("weight need to be square matrix.")
    weight[:] = np.identity(m)

def _gaussian_filler(weight, mu=0, sigma=1.0):
    """Initialize array using Gaussian distribution.
    """
    
    weight[:] = sigma * np.random.randn(weight.shape) + mu

def _constant_filler(bias):
    """Initialize all elements of array to be zero.
    """
    
    bias[:] = 0
      

_array_fillers = {'xavier': _xavier_filler,
                  'identity': _identity_filler,
                  'gaussian': _gaussian_filler,
                  'constant': _constant_filler}


