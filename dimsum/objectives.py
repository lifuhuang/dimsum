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
    
def _cross_entropy_loss(output, target):
    output = np.maximum(1e-17, output)
    return -np.sum(target * np.log(output), axis=-1).mean()

def _cross_entropy_derivative(output, target):    
    output = np.maximum(1e-17, output)
    return -(target / output) / (output.shape[0] if output.ndim > 1 else 1.0)
    
def _mean_square_loss(output, target):
    return ((output - target) ** 2).mean() / 2.0
    
def _mean_square_derivative(output, target):
    return (output - target) / output.size
    
_functions = {'cross_entropy': _cross_entropy_loss,
              'mean_square': _mean_square_loss}
_derivatives = {'cross_entropy': _cross_entropy_derivative,
                'mean_square': _mean_square_derivative}