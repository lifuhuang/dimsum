# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:41:04 2016

@author: lifu
"""

import numpy as np

class Regularizer(object):
    """Base class for all regularizers.
    """
    
    def __init__(self, lambda_):
        """Initialize a new instance of Regularizer.
        """
        
        self._lambda = lambda_
        
    def function(self, params):
        """Return penalty term for params.
        """
        
        raise NotImplementedError
    
    def derivative(self, params):
        """Return derivative of regularization term given params.
        """
        
        raise NotImplementedError
    

class L2(Regularizer):
    """Base class for all regularizers.
    """
    
    
    def function(self, params):
        """Return penalty term for params.
        """
        
        return self._lambda / 2.0 * np.sum(params ** 2)
    
    def derivative(self, params):
        """Return derivative of regularization term given params.
        """
        
        return self._lambda * params.copy()


class L1(Regularizer):
    """Base class for all regularizers.
    """
    
    
    def function(self, params):
        """Return penalty term for params.
        """
        
        return self._lambda * np.linalg.norm(params, ord=1)
    
    def derivative(self, params):
        """Return derivative of regularization term given params.
        """
        
        return self._lambda * ((params >= 0).astype(float) -
                                (params < 0).astype(float))