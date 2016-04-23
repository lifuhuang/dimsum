import sys
import nonlinearities
import initialization
import objectives
import itertools as it
import numpy as np

class ArrayPool(object):
    """Utility for storing and managing ndarrays in a centralized manner.
    """
    
    def __init__(self, array_shapes):
        """Initialize a new instance of ArrayPool.
        """
        self._names = array_shapes.keys()
        self._indices = {name: i for i, name in enumerate(self._names)}
        self._n_params = len(array_shapes)
        self._shapes = [array_shapes[name] for name in self._names]
        self._lens = map(np.prod, self._shapes)
        self._ends = np.cumsum(np.concatenate(([0], self._lens))).tolist()
        self._vec = np.zeros(np.sum(self._lens))
        self._views = []
        for i, name in enumerate(self._names):
            segment = self._vec[self._ends[i]:self._ends[i+1]]
            self._views.append(segment.reshape(self._shapes[i]))
            # support attribute-style access
            if not hasattr(self, name):
                setattr(self, name, self._views[i])
            else:
                raise ValueError('Parameter name %s has been reserved' % name)
                
    def flatten(self):
        """Return the flattened view of all parameters.
        """
        return self._vec
        
    def reset(self):
        """Reset all elements to zero.
        """
        self._vec[:] = 0        
        
    def names(self):
        """Return a list of parameter names.
        """
        return self._indices.keys()
        
    def __getitem__(self, key):
        idx = self._indices[key]
        return self._views[idx]

    def __setitem__(self, key, value):
        idx = self._indices[key]
        self._views[idx][:] = value
       
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
        if weight_filler in initialization.weight_fillers:
            self._weight_filler = initialization.weight_fillers[weight_filler]
        else:
            raise ValueError('Unrecognized weight_filler: %s.' % weight_filler)
            
        # bias filler
        if bias_filler in initialization.bias_fillers:
            self._bias_filler = initialization.bias_fillers[bias_filler]
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

class NeuralNetwork(object):
    """Base classes of all neural networks.
    """
    
    def __init__(self, objective='cross_entropy', derivative=None):
        """Initialize a new instance of NeuralNetwork.
        """
        
        self.params = None
        self.grads = None
        
        self._layers = []
        if objective in objectives.functions:
            self._objective = objectives.functions[objective]
            self._derivative = objectives.derivatives[objective]
        elif callable(objective):
            self._objective = objective
            self._derivative = derivative
        else:
            raise ValueError('%s is not a valid objective' % objective)
    
    def build(self):
        """Deploy the neural network and allocate memory to layers.
        """
        
        shapes = {}
        for l in self._layers:        
            shapes.update(l.param_shapes)
            
        self.params = ArrayPool(shapes)
        self.grads = ArrayPool(shapes)
        for layer in self._layers:
            layer.build()

    def predict(self, samples):
        """Return output of the network given input.
        """
        
        return self._forward_propagate(samples)
        
    def add(self, layer):
        """Add a layer to this network.
        """
        
        if isinstance(layer, Layer):
            self._layers.append(layer)
            layer.attached_to = self
        else:
            raise ValueError('A Layer instance should be passed in.')
    
    def grad_check(self, x, y, eps=1e-4, tol=1e-8,
           outfd=sys.stderr, skiplist=[]):
        """Check gradients on (x, y) using current params.
        """
        
        self.grads.reset()
        self._acc_gradients(x, y)
        
        for name in self.params.names():
            if name in skiplist: 
                continue
            print >> outfd, "Cheking dJ/d(%s)" % name,
            theta = self.params[name]
            grad_computed = self.grads[name]
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
                
        self.grads.reset()
        
    def _forward_propagate(self, message):
        msg = message
        for layer in self._layers:
            msg = layer.forward_propagate(msg)
        return msg
    
    def _compute_loss(self, message, targets):
        outputs = self._forward_propagate(message)
        return sum(self._objective(o, t) for o, t in it.izip(outputs, targets))
    
    def _back_propagate(self, message):
        msg = message
        for layer in self._layers[::-1]:
            msg = layer.back_propagate(msg)
        return msg
    
    def _acc_gradients(self, samples, targets):
        outputs = self._forward_propagate(samples)
        errors = self._derivative(outputs, targets)
        self._back_propagate(errors)

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add(DenseLayer(10, 50, name='layer1', nonlinearity='logistic'))
    nn.add(DenseLayer(50, 30, name='layer2', nonlinearity='tanh'))
    nn.add(DenseLayer(30, 20, name='layer3', nonlinearity='softmax'))
    nn.add(DenseLayer(20, 30, name='layer4', nonlinearity='identity'))    
    nn.add(DenseLayer(30, 30, name='layer5', nonlinearity='softmax'))
    nn.build()
    nn.grad_check(np.random.randn(10, 10), nonlinearities.softmax(np.random.randn(10, 30)))