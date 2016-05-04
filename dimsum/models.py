import sys

import numpy as np

from . import objectives

class NeuralNetwork(object):
    """Base classes of all neural networks.
    """
    
    
    def __init__(self, objective):
        """Initialize a new instance of NeuralNetwork.
        """
        
        self.layers = []
        self._objective = objectives.get(objective) or objective

    def predict(self, samples):
        """Return output of the network given input.
        """
        
        return self._forward_propagate(samples)
        
    def add(self, layer):
        """Add a layer to this network.
        """
        
        self.layers.append(layer)

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
                
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]
            self._reset_gradients()
            self._acc_gradients(x_batch, y_batch)
            for layer in self.layers:
                for param, grad in zip(layer.get_params(), layer.get_grads()):
                    optimizer.update(param, grad)
                    
            n_iters = i // batch_size + 1
            # call callbacks
            for callback, period in callbacks:
                if  n_iters % period == 0:
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
        
        params = []
        for layer in self.layers:
            params += layer.get_params()
        np.savez(path, *params)
    
    def load_params(self, path):
        """Load saved params from file.
        """
        
        params = np.load(path).items()
        params.sort()
        param_iter = iter(params)
        for layer in self.layers:
            for param in layer.get_params():
                param[:] = next(param_iter)[1]
        
    def grad_check(self, x, y, eps=1e-6, tol=1e-7,
           outfd=sys.stderr):
        """Check gradients on (x, y) using current params.
        """
        
        self._reset_gradients()
        self._acc_gradients(x, y)
        success = True
        
        for i, layer in enumerate(self.layers):
            print >> outfd, 'Layer %d:' % i
            for param, grad in zip(layer.get_params(), layer.get_grads()):
                grad_computed = grad
                grad_approx = np.zeros(param.shape)
                for idx, v in np.ndenumerate(param):
                    t = param[idx]
                    param[idx] = t + eps
                    Jplus  = self.compute_loss(x, y)
                    param[idx] = t - eps
                    Jminus = self.compute_loss(x, y)
                    param[idx] = t
                    grad_approx[idx] = (Jplus - Jminus) / (2 * eps)
                    
                grad_delta = np.linalg.norm(grad_approx - grad_computed)
                print >> outfd, "error norm = %.04g" % grad_delta,
                print >> outfd, ("[ok]" if grad_delta < tol else "[ERROR]")
                
                success &= (grad_delta < tol)
        self._reset_gradients()        
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

    def _reset_gradients(self):
        for layer in self.layers:
            layer.reset_grads()