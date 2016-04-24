# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:55:04 2016

@author: lifu
"""

import numpy as np

from . import utils

def get(name):
    """Return an activation function and its derivative according to name.
    """
    
    if name not in _functions:
        raise ValueError('%s is not a valid activation function' % name)    
    return _functions[name], _derivatives[name]

def _identity(x):
    """Return the value unchanged."""
    
    return x

def _tanh(x):
    """Return the tanh value of the array-like input. 
    """
    
    return np.tanh(x)

def _logistic(x):
    """Return the sigmoid value of the array-like input. 
    """
    
    return 1.0 / (1.0 + np.exp(-x))

def _relu(x):
    """Return the relu value of the array-like input.
    """
    
    return np.maximum(0, x)
    
def _softmax(x):
    """Return the Softmax value of the array-like input.
    """
    
    return utils.softmax(x)

_functions = {'identity': _identity, 
              'tanh': _tanh, 
              'logistic': _logistic,
              'relu': _relu,
              'softmax': _softmax}

def _logistic_derivative(z):
    """Return the derivative of logistic function given its function output.
    """
    
    return z * (1.0 - z)

def _tanh_derivative(z):
    """Return the derivative of tanh function given its function output.
    """
    
    return 1.0 - (z ** 2.0)

def _relu_derivative(z):
    """Return the derivative of ReLu function given its function output.
    """
    
    return (z > 0).astype(z.dtype)

def _identity_derivative(z):
    """Return the derivative of identity function given its function output.
    """
    
    return np.ones(z.shape)

def _softmax_derivative(z):
    """Return the derivative of softmax function given its function output.
    """
    
    return np.array([np.diag(p) - np.outer(p, p) for p in z])
    
_derivatives = {'identity': _identity_derivative,
                'tanh': _tanh_derivative,
                'logistic': _logistic_derivative,
                'relu': _relu_derivative,
                'softmax': _softmax_derivative}
        
