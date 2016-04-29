import sys

import numpy as np

from . import objectives
from . import optimizers
from . import utils
from .utils import ArrayPool
from .layers import Layer

class NeuralNetwork(object):
    """Base classes of all neural networks.
    """
    
    
    def __init__(self, objective, optimizer='sgd', derivative=None):
        """Initialize a new instance of NeuralNetwork.
        """
        
        self.params = None
        self.grads = None
        
        self.layers = []        
        self._optimizer = optimizers.get(optimizer)
        
        if callable(objective):
            if not callable(derivative):
                raise ValueError('No valid derivative is provided.')
            self._objective, self._derivative = objective, derivative
        else:
            self._objective, self._derivative = objectives.get(objective)
            
    def build(self):
        """Deploy the neural network and allocate memory to layers.
        """
        
        shapes = {}
        for l in self.layers:        
            shapes.update(l.param_shapes)
            
        self.params = ArrayPool(shapes)
        self.grads = ArrayPool(shapes)
        for layer in self.layers:
            layer.build()

    def predict(self, samples):
        """Return output of the network given input.
        """
        
        return self._forward_propagate(samples)
        
    def add(self, layer):
        """Add a layer to this network.
        """
        
        if isinstance(layer, Layer):
            self.layers.append(layer)
            layer.attach_to(self)
        else:
            raise ValueError('A Layer instance should be passed in.')

    def fit(self, training_set, callbacks=[], optimizer_args={}):
        """Train this model using x and y.
        """
            
        if isinstance(training_set, tuple):
            sample_iter = utils.random_iter(*training_set)
        else:
            sample_iter = training_set
            
        optimizer = self._optimizer(**optimizer_args)
        for x, y in sample_iter:
            self.grads.reset()
            self._acc_gradients(x, y)
            optimizer.update(self.params[:], self.grads[:])
            
            # call callbacks
            for callback, period in callbacks:
                if optimizer.n_iters % period == 0:
                    callback(self, optimizer)    
        
    def compute_loss(self, x, y_true):
        """Compute loss for a batch of samples.
        """
        y_pred = self._forward_propagate(x)
        return self._objective(y_pred, y_true)
    
    def grad_check(self, x, y, eps=1e-4, tol=1e-6,
           outfd=sys.stderr, skiplist=[]):
        """Check gradients on (x, y) using current params.
        """
        
        self.grads.reset()
        self._acc_gradients(x, y)
        success = True
        
        for name in self.params.names():
            if name in skiplist: 
                continue
            print >> outfd, "Cheking dJ/d(%s)" % name,
            theta = self.params.get(name)
            grad_computed = self.grads.get(name)
            grad_approx = np.zeros(theta.shape)
            for idx, v in np.ndenumerate(theta):
                t = theta[idx]
                theta[idx] = t + eps
                Jplus  = self.compute_loss(x, y)
                theta[idx] = t - eps
                Jminus = self.compute_loss(x, y)
                theta[idx] = t
                grad_approx[idx] = (Jplus - Jminus) / (2 * eps)
                
            grad_delta = np.linalg.norm(grad_approx - grad_computed)
            print >> outfd, "error norm = %.04g" % grad_delta,
            print >> outfd, ("[ok]" if grad_delta < tol else "[ERROR]")
            
            success &= (grad_delta < tol)
        self.grads.reset()        
        print >> outfd, "Result:", ("[Success]" if success else "[Failure]")
        return success
        
    def _forward_propagate(self, message):
        msg = message
        for layer in self.layers:
            msg = layer.forward_propagate(msg)
        return msg
    
    def _back_propagate(self, message):
        msg = message
        for layer in self.layers[::-1]:
            msg = layer.back_propagate(msg)
        return msg
    
    def _acc_gradients(self, x, y):
        y_pred = self._forward_propagate(x)
        errors = self._derivative(y_pred, y)
        self._back_propagate(errors)
