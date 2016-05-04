# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:09:59 2016

@author: lifu
"""

import numpy as np

class Objective(object):
    """Base class for all objective functions.
    """
    
    
    @staticmethod      
    def function(output, target):
        """Call objective function given output and target.
        """
        
        raise NotImplementedError
     
    @staticmethod  
    def derivative(output, target):
        """Call the derivative of objective function given output and target.
        """
        
        raise NotImplementedError
        
class CrossEntropy(Objective):
    """Categorical Cross-Entropy error.
    """
    
    
    @staticmethod    
    def function(output, target):
        """Call objective function given output and target.
        """
        
        assert(output.ndim == target.ndim == 2)
        output = np.clip(output, 1e-10, 1 - 1e-10)
        return -np.sum(target * np.log(output), axis=-1).mean()
        
    @staticmethod
    def derivative(output, target):
        """Call the derivative of objective function given output and target.
        """
        
        assert(output.ndim == target.ndim == 2)
        output = np.clip(output, 1e-10, 1 - 1e-10)
        return -(target / output) / output.shape[0]
        
class BinaryCrossEntropy(Objective):
    """Binary Cross-Entropy error.
    """
    
    
    @staticmethod    
    def function(output, target):
        """Call objective function given output and target.
        """
        
        assert(output.ndim == target.ndim == 2)
        output = np.clip(output, 1e-10, 1 - 1e-10)
        return -(target * np.log(output) + (1 - target) * 
                np.log(1 - output)).mean()
    
    @staticmethod
    def derivative(output, target):
        """Call the derivative of objective function given output and target.
        """
        
        assert(output.ndim == target.ndim == 2)
        output = np.clip(output, 1e-10, 1 - 1e-10)
        return -(target / output - (1 - target) / (1 - output)) / output.size
        
class MeanSquareError(Objective):
    """Mean Square Error.
    """
    
    
    @staticmethod    
    def function(output, target):
        """Call objective function given output and target.
        """
        
        assert(output.ndim == target.ndim == 2)
        return ((output - target) ** 2).mean() / 2.0
        
    @staticmethod
    def derivative(output, target):
        """Call the derivative of objective function given output and target.
        """
        
        assert(output.ndim == target.ndim == 2)
        return (output - target) / output.size
        
_objectives = {'mse': MeanSquareError,
               'cee': CrossEntropy,
               'bcee': BinaryCrossEntropy}
def get(name):
    """Return objective according to name.
    """
    
    return _objectives.get(name, None)