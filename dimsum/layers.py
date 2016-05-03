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
            
    def forward_propagate(self, msg):
        """Calculates activation of this layer given msg.
        """
        
        raise NotImplementedError
        
    def back_propagate(self, msg):
        """Updates delta, gradients, and error message to lower layers.
        """
        
        raise NotImplementedError

    def compute_reg_loss(self):
        """Return penalty from regularization.
        """
        
        raise NotImplementedError
    
    def build(self):
        """Deploy this Layer and obtain actual memory.
        """
    
        raise NotImplementedError

class Input(Layer):
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

    def compute_reg_loss(self):
        """Return penalty from regularization.
        """
        
        return 0.0
        
    def build(self):
        """Deploy this Layer and obtain actual memory.
        """
    
        pass

class Dense(Layer):
    """A simple full-connected feedforward layer.
    """
    
    
    def __init__(self, size, name=None,
                 W_init='xavier', b_init='constant', 
                 W_regularizer=None, b_regularizer=None,
                 activation=activations.Identity, bias=True):
        """Initialize a new instance of DenseLayer.
        """
        
        # initialize base class
        super(type(self), self).__init__(size, name)
        
        self._bias = bias
        
        # parameters
        self.W = None
        self.dW = None
        if self._bias:        
            self.b = None  
            self.db = None
        
        # weight filler
        if callable(W_init):
            self._W_init = W_init
        else:
            self._W_init = initializations.get(W_init)
            
        # bias filler
        if callable(b_init):
            self._b_init = b_init
        else:
            self._b_init = initializations.get(b_init)
        
        self._activation = activation
        
        # regularizers
        self._W_reg = W_regularizer
        self._b_reg = b_regularizer
            
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
        if self._bias:
            self.param_shapes['%s_b' % self.name] = (self.size,)
                
    def forward_propagate(self, msg):
        """Calculates activation of this layer given msg.
        """
        
        self._input = msg
        z = np.dot(msg, self.W)
        if self._bias:
            z += self.b
        self._output = self._activation.function(z)
        return self._output
        
    def back_propagate(self, msg):
        """Updates delta, gradients, and error message to lower layers.
        """
        
        dfs = self._activation.derivative(self._output)
        if dfs.ndim == msg.ndim:
            deltas = msg * dfs
        else:
            deltas = np.array([np.dot(m, d.T) for m, d in it.izip(msg, dfs)])
            
        self.dW += np.dot(self._input.T, deltas) 
        if self._W_reg:
            self.dW += self._W_reg.derivative(self.W)
            
        if self._bias:
            self.db += np.sum(deltas, axis=0)
            if self._b_reg:
                self.db += self._b_reg.derivative(self.b)
                
        return np.dot(deltas, self.W.T)

    def compute_reg_loss(self):
        """Return penalty from regularization term.
        """
        
        loss = 0
        if self._W_reg is not None:
            loss += self._W_reg.function(self.W)
        if self._bias and self._b_reg is not None:
            loss += self._b_reg.function(self.b)
        return loss
        
    def build(self):        
        """Deploy this Layer and obtain actual memory.
        """
        
        self.W = self.attached_to.params.get('%s_W' % self.name)
        self._W_init(self.W)
        self.dW = self.attached_to.grads.get('%s_W' % self.name)
        
        if self._bias:
            self.b = self.attached_to.params.get('%s_b' % self.name)
            self._b_init(self.b)
            self.db = self.attached_to.grads.get('%s_b' % self.name)
