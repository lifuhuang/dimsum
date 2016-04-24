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
    return -np.dot(target.T, np.log(output))

def _cross_entropy_derivative(output, target):
    return -target / output
    
_functions = {'cross_entropy': _cross_entropy_loss}
_derivatives = {'cross_entropy': _cross_entropy_derivative}