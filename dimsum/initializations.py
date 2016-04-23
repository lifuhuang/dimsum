# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:10:50 2016

@author: lifu
"""
import math
import numpy as np

def get_weight_filler(name):
    """Return a weight filler according to name.
    """
        
    if name not in _weight_fillers:
        raise ValueError('%s is not a valid weight filler' % name)
    return _weight_fillers[name]
    
    
def get_bias_filler(name):
    """Return a weight filler according to name.
    """
    
    if name not in _bias_fillers:
        raise ValueError('%s is not a valid bias filler' % name)
    return _bias_fillers[name]


def _xavier_weight_filler(weight):
    """Initialize weight matrix with Xavier method.
    """
    
    r = math.sqrt(6.0 / (weight.shape[0] + weight.shape[1]))
    weight[:] = np.random.uniform(-r, r, weight.shape)
    
def _identity_weight_filler(weight):
    """Initialize weight to a identity matrix.
    """
    
    m, n = weight.shape
    if m != n:    
        raise ValueError("weight need to be square matrix.")
    weight[:] = np.eye(m, m)

def _gaussian_weight_filler(weight, mu=0, sigma=1.0):
    """Initialize weight matrix using Gaussian distribution.
    """
    
    weight[:] = sigma * np.random.randn(weight.shape) + mu

_weight_fillers = {'xavier': _xavier_weight_filler,
                  'identity': _identity_weight_filler,
                  'gaussian': _gaussian_weight_filler}

def _constant_bias_filler(bias):
    """Initialize all elements of bias to be zero.
    """
    
    bias[:] = 0
      
_bias_fillers = {'constant': _constant_bias_filler}
