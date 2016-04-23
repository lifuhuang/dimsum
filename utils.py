# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:07:39 2016

@author: lifu
"""

import matplotlib.pyplot as plt 
import numpy as np

def plot_matrix(x, **kwargs):
    """Plot the contour of a matrix.
    """
    
    indices = np.indices(x.shape)
    plt.contourf(indices[0], indices[1], x, **kwargs)   

def make_onehot(i, n):
    """Make an array with its ith element being one, others zero.
    """
    
    y = np.zeros(n)
    y[i] = 1
    return y