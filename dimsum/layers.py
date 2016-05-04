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
    
    
    def __init__(self, input_dim, output_dim):
        """Initialize a new instance of Layer.
        
        Since Layer is a abstract class, this method should only be called by
        its derived classes.
        """
        
        self.input_dim = input_dim
        self.output_dim = output_dim
            
    def forward_propagate(self, msg):
        """Calculate output of this layer given msg.
        """
        
        raise NotImplementedError
        
    def back_propagate(self, msg):
        """Update gradients, and propagate error message to lower layers.
        """
        
        raise NotImplementedError

    def compute_reg_loss(self):
        """Return penalty from regularization.
        """
        
        raise NotImplementedError
    
    def get_params(self):
        """Return a tuple of parameters.
        """
        
        raise NotImplementedError
        
    def get_grads(self):
        """Return a tuple of gradients.
        """
        
        raise NotImplementedError
    
    def reset_grads(self):
        """Reset all gradients to zeros.
        """
        
        raise NotImplementedError

class Input(Layer):
    """Dumb layer with no parameters.
        """
    
    def __init__(self, dim):
        """
        """
        
        super(type(self), self).__init__(dim, dim)
        
    def forward_propagate(self, msg):
        """Return input message without making any changes.
        """
        
        return msg
        
    def back_propagate(self, msg):
        """Return received message without making any changes.
        """
        
        return msg

    def compute_reg_loss(self):
        """Return zero.
        """
        
        return 0.0
    
    def get_params(self):
        """Return an empty tuple.
        """
        
        return tuple()
        
    def get_grads(self):
        """Return an empty tuple.
        """
        
        return tuple()
    
    def reset_grads(self):
        """Do nothing.
        """
        
        pass

class Dense(Layer):
    """A simple full-connected feedforward layer.
    """
    
    
    def __init__(self, input_dim, output_dim, 
                 W_init='xavier', b_init='constant', 
                 activation='identity', 
                 W_regularizer=None, b_regularizer=None, bias=True):
        """Initialize a new instance of DenseLayer.
        """
        
        super(type(self), self).__init__(input_dim, output_dim)
        
        # public attributes
        self.bias = bias
        self.W = np.empty((self.input_dim, self.output_dim))
        self.b = np.empty(self.output_dim) if self.bias else None
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape) if self.bias else None
        
        self._W_init = initializations.get(W_init) or W_init
        self._W_init(self.W)
        self._b_init = initializations.get(b_init) or b_init
        self._b_init(self.b)
        
        self._activation = activations.get(activation) or activation
        
        self._W_reg = W_regularizer
        self._b_reg = b_regularizer
            
        # cache for input/output
        self._input = None
        self._output = None    
    
    def get_params(self):
        """Return a tuple of parameters.
        """
        
        return self.W, self.b
        
    def get_grads(self):
        """Return a tuple of gradients.
        """
        
        return self.dW, self.db
    
    def reset_grads(self):
        """Reset dW and db to zeros.
        """
        
        self.dW[:] = 0
        self.db[:] = 0
    
    def forward_propagate(self, msg):
        """Calculate output of this layer given msg.
        """
        
        self._input = msg
        z = np.dot(msg, self.W)
        if self.bias:
            z += self.b
        self._output = self._activation.function(z)
        return self._output
        
    def back_propagate(self, msg):
        """Update gradients, and propagate error message to lower layers.
        """
        
        dfs = self._activation.derivative(self._output)
        if dfs.ndim == msg.ndim:
            deltas = msg * dfs
        else:
            deltas = np.array([np.dot(m, d.T) for m, d in it.izip(msg, dfs)])
            
        self.dW += np.dot(self._input.T, deltas) 
        if self._W_reg:
            self.dW += self._W_reg.derivative(self.W)
            
        if self.bias:
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
        if self.bias and self._b_reg is not None:
            loss += self._b_reg.function(self.b)
        return loss
        