# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:55:04 2016

@author: lifu
"""

import numpy as np

def get_function(name):
    """Return an activation function according to name.
    """
    
    if name not in _functions:
        raise ValueError('%s is not a valid activation function' % name)    
    return _functions[name]

def get_derivative(name):    
    """Return the derivative of activation function according to name.
    """
    
    if name not in _derivatives:
        raise ValueError('%s is not a valid activation function' % name)
    return _derivatives[name]

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
    
    x_exp = np.exp(x - np.max(x, axis = -1, keepdims = True))
    return x_exp / np.sum(x_exp, axis = -1, keepdims = True)

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
        