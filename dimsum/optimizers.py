# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:17:32 2016

@author: lifu
"""


import numpy as np

class Optimizer(object):
    """Abstract base class for all Optimizers.
    
    This class is intended to be inherited. Should not be use as a real
    optimizer.
    """

    
    def __init__(self):
        """Initialize a new Optimizer instance.
        """
        
        raise NotImplementedError
    
    def reset(self):
        """Reset optimizer to initial state.
        
        This function is intended to be called everytime before training.
        """
        
        raise NotImplementedError
        
    def update(self, param, grad):
        """Update parameter given their gradient.
        """
        
        raise NotImplementedError

class SGD(Optimizer):    
    """Stochastic Gradient Descent optimizer.
    """

    
    def __init__(self, learning_rate, decay_rate=1.0, decay_period=None):
        """Initialize a new SgdOptimizer instance.
        """
        
        self.init_lr = learning_rate
        self.decay_rate = decay_rate
        self.decay_period = decay_period
    
    @property
    def learning_rate(self):
        """Return current learning rate.
        """
        
        if self.decay_period is None:
            return self.init_lr
        else:
            return self.init_lr * (self.decay_rate ** 
                                    (self.n_iters // self.decay_period))
        
    def update(self, param, grad):
        """Update parameter given their gradient.
        """
        
        param -= self.learning_rate * grad
        

class Adagrad(Optimizer):    
    """Stochastic Gradient Descent optimizer.
    """

    
    def __init__(self, learning_rate):
        """Initialize a new SgdOptimizer instance.
        """
        
        self.learning_rate = learning_rate
        self._cache = {}
        
    def update(self, param, grad):
        """Update parameter given their gradient.
        """
        
        param_id = id(param)
        cache = self._cache.get(param_id, None)
        if cache is None:
            cache = grad ** 2
            self._cache[param_id] = cache
        else:
            cache += grad ** 2
        param -= self.learning_rate * grad / np.sqrt(cache + 1e-8)
