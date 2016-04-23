# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:17:39 2016

@author: lifu
"""

import itertools as it
import numpy as np
import initializations
import nonlinearities

class Layer(object):
    """Base class for all kinds of layers.
    """
    
    
    def __init__(self, name=None):
        """Initialize a new instance of Layer.
        
        Since Layer is a abstract class, this method should only be called by
        its derived classes.
        """
        
        self.param_shapes = {}
        self.attached_to = None
        self.name = id(self) if name is None else name
    
    def forward_propagate(self, msg):
        """Calculates activation of this layer given msg.
        """
        
        raise NotImplementedError
        
    def back_propagate(self, msg, update=True):
        """Updates delta, gradients, and error message to lower layers.
        """
        
        raise NotImplementedError
        
    def builds(self):
        """Deploy this Layer and obtain actual memory.
        """
    
        raise NotImplementedError
        
    def get_config(self):
        """Return a dict containing information of this Layer.
        
        This is used for serialization, thus derived classes should override 
        this by returning a dict containing their own information as well as 
        that of base class obtained by calling this method.
        """
        
        return {'params_shapes': self.param_shapes,
                'attached_to': self.attached_to,
                'name': self.name}

class DenseLayer(Layer):
    """A simple full-connected feedforward layer.
    """
    
    
    def __init__(self, fan_in, fan_out, nonlinearity='logistic', 
                 weight_filler='xavier', bias_filler='constant', **kwargs):
        """Initialize a new instance of DenseLayer.
        """
        
        # initialize base class
        super(type(self), self).__init__(**kwargs)
        
        # set size info
        self.fan_in = fan_in
        self.fan_out = fan_out
        
        # parameters
        self.W = None
        self.b = None  
        self.dW = None
        self.db = None
        self.param_shapes['%s_W' % self.name] = (fan_in, fan_out)
        self.param_shapes['%s_b' % self.name] = (fan_out,)
        
        # weight filler
        if weight_filler in initializations.weight_fillers:
            self._weight_filler = initializations.weight_fillers[weight_filler]
        else:
            raise ValueError('Unrecognized weight_filler: %s.' % weight_filler)
            
        # bias filler
        if bias_filler in initializations.bias_fillers:
            self._bias_filler = initializations.bias_fillers[bias_filler]
        else:
            raise ValueError('Unrecognized bias_filler: %s.' % bias_filler)
        
        # nonlinearity and derivative
        if nonlinearity in nonlinearities.functions:  
            self._nonlinearity = nonlinearities.functions[nonlinearity]
            self._derivative = nonlinearities.derivatives[nonlinearity]
        else:
            raise ValueError('Unrecognized nonlinearity: %s.' % nonlinearity)
        
        # cache of input/output
        self._input = None
        self._output = None    

    def get_config(self):
        """Return a dict containing information of this Layer.
        """
        
        return {'params': self.params,
                'attached_to': self.attached_to,
                'name': self.name}
                
    def forward_propagate(self, msg):
        """Calculates activation of this layer given msg.
        """
        
        self._input = msg
        self._output = self._nonlinearity(np.dot(msg, self.W) + self.b)
        return self._output
        
    def back_propagate(self, msg):
        """Updates delta, gradients, and error message to lower layers.
        """
        
        dfs = self._derivative(self._output)
        if dfs.ndim == msg.ndim:
            deltas = msg * dfs
        else:
            deltas = np.array([np.dot(m, d.T) for m, d in it.izip(msg, dfs)])
            
        self.dW += np.dot(self._input.T, deltas)
        self.db += np.sum(deltas, axis=0)
        return np.dot(deltas, self.W.T)

    def build(self):        
        """Deploy this Layer and obtain actual memory.
        """
        
        self.W = self.attached_to.params['%s_W' % self.name]
        self._weight_filler(self.W)
        self.dW = self.attached_to.grads['%s_W' % self.name]
        
        self.b = self.attached_to.params['%s_b' % self.name]
        self._bias_filler(self.b)
        self.db = self.attached_to.grads['%s_b' % self.name]
