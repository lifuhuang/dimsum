import sys
import itertools as it

import numpy as np

from .import objectives
from .utils import ArrayPool
from .layers import Layer

class NeuralNetwork(object):
    """Base classes of all neural networks.
    """
    
    
    def __init__(self, objective='cross_entropy', derivative=None):
        """Initialize a new instance of NeuralNetwork.
        """
        
        self.params = None
        self.grads = None
        
        self.layers = []
        
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
            layer.attach_to(self)
        else:
            raise ValueError('A Layer instance should be passed in.')

    def fit(self, x, y, optimizer):
        pass
    
    def grad_check(self, x, y, eps=1e-4, tol=1e-8,
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
                Jplus  = self._compute_loss(x, y)
                theta[idx] = t - eps
                Jminus = self._compute_loss(x, y)
                theta[idx] = t
                grad_approx[idx] = (Jplus - Jminus) / (2 * eps)
                
            grad_delta = np.linalg.norm(grad_approx - grad_computed)
            print >> outfd, "error norm = %.04g" % grad_delta,
            print >> outfd, ("[ok]" if grad_delta < tol else "[ERROR]")
            success &= (grad_delta < tol)
                
        self.grads.reset()
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
        
    def _compute_loss(self, message, targets):
        outputs = self._forward_propagate(message)
        return sum(self._objective(o, t) for o, t in it.izip(outputs, targets))
    
    def _acc_gradients(self, samples, targets):
        outputs = self._forward_propagate(samples)
        errors = self._derivative(outputs, targets)
        self._back_propagate(errors)
