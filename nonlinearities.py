# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:55:04 2016

@author: lifu
"""

import numpy as np

def identity(x):
    """Return the value unchanged."""
    
    return x

def tanh(x):
    """Return the tanh value of the array-like input. 
    """
    
    return np.tanh(x)

def logistic(x):
    """Return the sigmoid value of the array-like input. 
    """
    
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    """Return the relu value of the array-like input.
    """
    
    return np.maximum(0, x)
    
def softmax(x):
    """Return the Softmax value of the array-like input.
    """
    
    x_exp = np.exp(x - np.max(x, axis = -1, keepdims = True))
    return x_exp / np.sum(x_exp, axis = -1, keepdims = True)

functions = {'identity': identity, 
             'tanh': tanh, 
             'logistic': logistic,
             'relu': relu,
             'softmax': softmax}

def quick_logistic_derivative(z):
    """Return the derivative of logistic function given its function output.
    """
    
    return z * (1.0 - z)

def quick_tanh_derivative(z):
    """Return the derivative of tanh function given its function output.
    """
    
    return 1.0 - (z ** 2.0)

def quick_relu_derivative(z):
    """Return the derivative of ReLu function given its function output.
    """
    
    return (z > 0).astype(z.dtype)

def quick_identity_derivative(z):
    """Return the derivative of identity function given its function output.
    """
    
    return np.ones(z.shape)

def quick_softmax_derivative(z):
    """Return the derivative of softmax function given its function output.
    """
    
    return np.array([np.diag(p) - np.outer(p, p) for p in z])
    
derivatives = {'identity': quick_identity_derivative,
               'tanh': quick_tanh_derivative,
               'logistic': quick_logistic_derivative,
               'relu': quick_relu_derivative,
               'softmax': quick_softmax_derivative}
        