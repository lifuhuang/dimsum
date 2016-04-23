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

class ArrayPool(object):
    """Utility for storing and managing ndarrays in a centralized manner.
    """
    
    def __init__(self, array_shapes):
        """Initialize a new instance of ArrayPool.
        """
        self._names = array_shapes.keys()
        self._indices = {name: i for i, name in enumerate(self._names)}
        self._n_params = len(array_shapes)
        self._shapes = [array_shapes[name] for name in self._names]
        self._lens = map(np.prod, self._shapes)
        self._ends = np.cumsum(np.concatenate(([0], self._lens))).tolist()
        self._vec = np.zeros(np.sum(self._lens))
        self._views = []
        for i, name in enumerate(self._names):
            segment = self._vec[self._ends[i]:self._ends[i+1]]
            self._views.append(segment.reshape(self._shapes[i]))
            # support attribute-style access
            if not hasattr(self, name):
                setattr(self, name, self._views[i])
            else:
                raise ValueError('Parameter name %s has been reserved' % name)
                
    def flatten(self):
        """Return the flattened view of all parameters.
        """
        return self._vec
        
    def reset(self):
        """Reset all elements to zero.
        """
        self._vec[:] = 0        
        
    def names(self):
        """Return a list of parameter names.
        """
        return self._indices.keys()
        
    def __getitem__(self, key):
        idx = self._indices[key]
        return self._views[idx]

    def __setitem__(self, key, value):
        idx = self._indices[key]
        self._views[idx][:] = value
       