import sys

import numpy as np

from .utils import ArrayPool
from .layers import Layer

class NeuralNetwork(object):
    """Base classes of all neural networks.
    """
    
    
    def __init__(self, objective):
        """Initialize a new instance of NeuralNetwork.
        """
        
        self.params = None
        self.grads = None
        
        self.layers = []      
        
        self._objective = objective
        
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

    def fit(self, x, y, optimizer, n_epochs=5, batch_size=32,
            randomized=True, callbacks=[]):
        """Train this model using x and y.
        """
        
        assert x.shape[0] == y.shape[0]
        
        # wrap current model into args passed to callbacks
        model = self
        epoch_size = x.shape[0]
        
        for i in xrange(0, n_epochs * epoch_size, batch_size):
            if randomized:
                batch_indices = np.random.randint(0, epoch_size, batch_size)
            else:
                st = i % epoch_size
                batch_indices = np.arange(st, st + batch_size)
                
            self.grads.reset()
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]
            self._acc_gradients(x_batch, y_batch)
            optimizer.update(self.params[:], self.grads[:])
            
            # call callbacks
            for callback, period in callbacks:
                if optimizer.n_iters % period == 0:
                    callback(locals())    
        
    def compute_loss(self, x, y_true):
        """Compute loss for a batch of samples.
        """
        
        reg_loss = 0.0
        msg = x
        for layer in self.layers:
            msg = layer.forward_propagate(msg)
            reg_loss += layer.compute_reg_loss()
        return reg_loss + self._objective.function(msg, y_true)
    
    def save_params(self, path):
        """Save params to file.
        """
        
        np.save(path, self.params)
    
    def load_params(self, path):
        """Load saved params from file.
        """
        
        self.params[:] = np.load(path)
        
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
        errors = self._objective.derivative(y_pred, y)
        self._back_propagate(errors)
