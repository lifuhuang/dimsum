# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:55:04 2016

@author: lifu
"""

import numpy as np

from . import utils

class Activation(object):
    """Base class for all activation functions and their gradients.
    """

        
    def function(self, x):
        """Call activation function given input x.
        """
        
        raise NotImplementedError
        
    def derivative(self, y):
        """Call the derivative of activation function given function output y.
        """
        
        raise NotImplementedError

class ReLU(Activation):
    """Rectified Linear Unit.
    """
    
    @staticmethod
    def function(x):
        """Return the relu value of the array-like input.
        """
        
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(y):
        """Return the derivative of ReLu function given its function output.
        """
    
        return (y > 0).astype(y.dtype)

class Tanh(Activation):
    """Hyperbolic function.
    """
    
    @staticmethod
    def function(x):
        """Return the tanh value of the array-like input. 
        """
    
        return np.tanh(x)
        
    @staticmethod
    def derivative(y):
        """Return the derivative of tanh function given its function output.
        """
    
        return 1.0 - (y ** 2.0)    
        
class Softmax(Activation):
    """Softmax function.
    """

    @staticmethod
    def function(x):
        """Return the Softmax value of the array-like input.
        """
    
        return utils.softmax(x)   
        
    @staticmethod
    def derivative(y):
        """Return the derivative of softmax function given its function output.
        """
        
        return np.array([np.diag(p) - np.outer(p, p) for p in y])

class Sigmoid(Activation):
    """Logistic function.
    """
    
    @staticmethod
    def function(x):
        """Return the derivative of logistic function given its function output.
        """
    
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def derivative(y):
        """Return the derivative of logistic function given its function output.
        """
        
        return y * (1.0 - y)

class Identity(Activation):
    """Dummy activation function.
    """
    
    @staticmethod
    def function(x):
        """Return the input unchanged
        """
    
        return x
    
    @staticmethod
    def derivative(y):
        """Return the derivative of identity function.
        """
        
        return np.ones(y.shape)

_activations = {'relu': ReLU,
                'identity': Identity,
                'sigmoid': Sigmoid,
                'softmax': Softmax,
                'tanh': Tanh}

def get(name):
    """Return activation according to name.
    """
    
    return _activations.get(name, None)