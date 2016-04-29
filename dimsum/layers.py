# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:17:39 2016

@author: lifu
"""

import itertools as it

import numpy as np

from . import initializations
from . import activations

class Layer(object):
    """Base class for all kinds of layers.
    """
    
    
    def __init__(self, size, name=None):
        """Initialize a new instance of Layer.
        
        Since Layer is a abstract class, this method should only be called by
        its derived classes.
        """
        
        self.param_shapes = {}
        self.size = size
        self.attached_to = None
        self.name = id(self) if name is None else name
        
    def attach_to(self, model):
        """Attach this layer to a neural network model.
        
        This method can be overridden to specify operations to do when a layer
        is added to a neural network model, such as setting param_shapes. Sub-
        classes are supposed to call this in base class before doing their own
        tasks.
        """
        
        if self not in model.layers:
            raise ValueError('Layer is not in model %s\'s layer list.' % model)
        self.attached_to = model
            
    def get_config(self):
        """Return a dict containing information of this Layer.
        
        This is used for serialization,
        thus derived classes should override 
        this by returning a dict containing their own information as well as 
        that of base class obtained by calling this method.
        """
        
        return {'params_shapes': self.param_shapes,
                'size': self.size,
                'attached_to': self.attached_to,
                'name': self.name}
                
    def forward_propagate(self, msg):
        """Calculates activation of this layer given msg.
        """
        
        raise NotImplementedError
        
    def back_propagate(self, msg):
        """Updates delta, gradients, and error message to lower layers.
        """
        
        raise NotImplementedError
        
    def build(self):
        """Deploy this Layer and obtain actual memory.
        """
    
        raise NotImplementedError

class InputLayer(Layer):
    """Dumb layer with no parameters.
        """
    
    def forward_propagate(self, msg):
        """Calculates activation of this layer given msg.
        """
        
        return msg
        
    def back_propagate(self, msg):
        """Updates delta, gradients, and error message to lower layers.
        """
        
        return msg
        
    def build(self):
        """Deploy this Layer and obtain actual memory.
        """
    
        pass

class DenseLayer(Layer):
    """A simple full-connected feedforward layer.
    """
    
    
    def __init__(self, size, name=None,
                 weight_filler='xavier', bias_filler='constant', 
                 activation='logistic', derivative=None):
        """Initialize a new instance of DenseLayer.
        """
        
        # initialize base class
        super(type(self), self).__init__(size, name)
        
        # parameters
        self.W = None
        self.b = None  
        self.dW = None
        self.db = None
        
        # weight filler
        if callable(weight_filler):
            self._weight_filler = weight_filler
        else:
            self._weight_filler = initializations.get(weight_filler)
            
        # bias filler
        if callable(bias_filler):
            self._bias_filler = bias_filler
        else:
            self._bias_filler = initializations.get(bias_filler)
        
        # nonlinearity and derivative
        if callable(activation):
            if not callable(derivative):
                raise ValueError('No valid derivative is provided.')
            self._activation, self._derivative = activation, derivative
        else:
            self._activation, self._derivative = activations.get(activation)
        
        # cache for input/output
        self._input = None
        self._output = None    
    
    def attach_to(self, model):
        """Attach this layer to a neural network model.
        """
        
        
        super(type(self), self).attach_to(model)            
        index = model.layers.index(self)
        if index == 0:
            raise ValueError('Input layer is needed before DenseLayer.')
        prev_layer = model.layers[index - 1]
        self.param_shapes['%s_W' % self.name] = (prev_layer.size, self.size)
        self.param_shapes['%s_b' % self.name] = (self.size,)
        
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
        self._output = self._activation(np.dot(msg, self.W) + self.b)
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
        
        self.W = self.attached_to.params.get('%s_W' % self.name)
        self._weight_filler(self.W)
        self.dW = self.attached_to.grads.get('%s_W' % self.name)
        
        self.b = self.attached_to.params.get('%s_b' % self.name)
        self._bias_filler(self.b)
        self.db = self.attached_to.grads.get('%s_b' % self.name)
