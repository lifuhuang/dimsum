# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:09:59 2016

@author: lifu
"""

import numpy as np

def get(name):
    """Return an objective function and its derivative according to name.
    """
    if name not in _functions:
        raise ValueError('%s is not a valid objective.' % name)
    return _functions[name], _derivatives[name]
    
def _categorical_crossentropy_error(output, target):
    assert(output.ndim == target.ndim == 2)
    output = np.clip(1e-10, 1 - 1e-10)
    return -np.sum(target * np.log(output), axis=-1).mean()

def _categorical_crossentropy_derivative(output, target):    
    assert(output.ndim == target.ndim == 2)
    output = np.clip(1e-10, 1 - 1e-10)
    return -(target / output) / output.shape[0]

def _binary_crossentropy_error(output, target):
    assert(output.ndim == target.ndim == 2)
    output = np.clip(1e-10, 1 - 1e-10)
    return -np.sum(target * np.log(output) + (1 - target) * 
            np.log(1 - output)).mean()

def _binary_crossentropy_derivative(output, target):
    assert(output.ndim == target.ndim == 2)
    output = np.clip(1e-10, 1 - 1e-10)
    return (target / output - (1 - target) / (1 - output)) / output.size
    
def _mean_square_error(output, target):
    assert(output.ndim == target.ndim == 2)
    return ((output - target) ** 2).mean() / 2.0
    
def _mean_square_derivative(output, target):
    assert(output.ndim == target.ndim == 2)
    return (output - target) / output.size
    
_functions = {'cce': _categorical_crossentropy_error,
              'categorical_crossentropy': _categorical_crossentropy_error,
              'mse': _mean_square_error,
              'mean_square': _mean_square_error,
              'bce': _binary_crossentropy_error,
              'binary_crossentropy': _binary_crossentropy_error}
              
_derivatives = {'cce': _categorical_crossentropy_derivative,
              'categorical_crossentropy': _categorical_crossentropy_derivative,
              'mse': _mean_square_derivative,
              'mean_square': _mean_square_derivative,
              'bce': _binary_crossentropy_derivative,
              'binary_crossentropy': _binary_crossentropy_derivative}