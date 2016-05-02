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
        
        self.n_iters = 0
    
    def reset(self):
        """Reset optimizer to initial state.
        
        This function is intended to be called everytime before training.
        """
        
        self.n_iters = 0
        
    def update(self, params, grads):
        """Update parameters given their gradients.
        """
        
        self.n_iters += 1

class SGD(Optimizer):    
    """Stochastic Gradient Descent optimizer.
    """

    
    def __init__(self, learning_rate, decay_rate=1.0, decay_period=None):
        """Initialize a new SgdOptimizer instance.
        """
        
        super(type(self), self).__init__()
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
        
    def update(self, params, grads):
        """Update parameters given their gradients.
        """
        
        super(type(self), self).update(params, grads)
        params -= self.learning_rate * grads
        

class Adagrad(Optimizer):    
    """Stochastic Gradient Descent optimizer.
    """

    
    def __init__(self, learning_rate):
        """Initialize a new SgdOptimizer instance.
        """
        
        super(type(self), self).__init__()
        self.learning_rate = learning_rate
        self._cache = None
        
    def update(self, params, grads):
        """Update parameters given their gradients.
        """
        
        super(type(self), self).update(params, grads)
        if self._cache is None:
            self._cache = np.zeros(params.shape)
            
        self._cache += grads ** 2
        params -= self.learning_rate * grads / np.sqrt(self._cache + 1e-8)

