# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:07:39 2016

@author: lifu
"""

import matplotlib.pyplot as plt 
import numpy as np

def softmax(x):
    """Return the Softmax value of the array-like input.
    """
    
    x_exp = np.exp(x - np.max(x, axis = -1, keepdims = True))
    return x_exp / np.sum(x_exp, axis = -1, keepdims = True)

def plot_matrix(x, **kwargs):
    """Plot the contour of a matrix.
    """
    
    indices = np.indices(x.shape)
    plt.contourf(indices[0], indices[1], x, **kwargs)   

def make_onehot(i, n):
    """Make an array with its ith element being one, others zero.
    """
    
    y = np.zeros(n) 
    y[i] = 1.0
    return y
    
def make_onehots(indices, size):
    """Make a 2-d array with only one element being one for each row.
    """
    
    y = np.zeros(size) 
    y[np.arange(size[0]), indices] = 1.0
    return y

def random_iter(x, y, batch_size=20, n_epochs=5):
    """Iterate over a set of samples randomly.
    """
    
    assert x.shape[0] == y.shape[0]
    epoch_size = x.shape[0]
    for i in xrange(0, n_epochs * epoch_size, batch_size):
        indices = np.random.randint(0, epoch_size, batch_size)
        yield (x[indices], y[indices])

def sequential_iter(x, y, batch_size=20, n_epochs=5):
    """Iterate over a set of samples sequentially.
    """
    
    assert x.shape[0] == y.shape[0]
    epoch_size = x.shape[0]
    for i in xrange(0, n_epochs * epoch_size, batch_size):
        indices = np.arange(i % epoch_size, (i + batch_size) % epoch_size)
        yield (x[indices], y[indices])
    
class ArrayPool(object):
    """Utility for storing and managing ndarrays in a centralized manner.
    
    ArrayPool provides a centralized interface for managing a set of arrays, 
    and supports all ndarray operations by redirecting them to the underlying
    big ndarray.
    """
    
    
    def __init__(self, array_shapes):
        """Initialize a new instance of ArrayPool.
        """
        
        self._names = array_shapes.keys()
        self._indices = {name: i for i, name in enumerate(self._names)}
        self._n_params = len(array_shapes)
        self._shapes = [array_shapes[name] for name in self._names]
        self._lens = map(np.prod, self._shapes)
        self._ends = np.cumsum(np.array([0] + self._lens)).tolist()
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
                
    def __getattr__(self, key):
        """Redirect attribute access to underlying ndarray.
        """
        
        return getattr(self._vec, key)
        
    def __getitem__(self, key):
        """Redirect indexing to underlying ndarray.
        """
        
        return self._vec.__getitem__(key)
    
    def __setitem__(self, key, value):
        """Redirect indexing to underlying ndarray.
        """
        
        return self._vec.__setitem__(key, value)
    
    def __len__(self):
        """Redirect len operation to underlying ndarray.
        """
        
        return len(self._vec)        
    
    def __iter__(self):
        """Redirect len operation to underlying ndarray.
        """
        
        return iter(self._vec)
    
    def reset(self):
        """Reset all elements to zero.
        """
        self._vec[:] = 0        
        
    def names(self):
        """Return a list of parameter names.
        """
        return self._indices.keys()
        
    def get(self, name):
        """Return view of array with given name.
        """
        
        idx = self._indices[name]
        return self._views[idx]
       
