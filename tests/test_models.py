# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:27:21 2016

@author: lifu
"""

import numpy as np

from dimsum.layers import *
from dimsum.activations import *
from dimsum.objectives import *
from dimsum.optimizers import *
from dimsum.regularizers import *
from dimsum.models import NeuralNetwork
from dimsum import utils
from dimsum import callbacks

class TestNeuralNetwork:
    """Test class for NeuralNetwork.
    """
    

    def test_grad_check(self):
        """Perform gradient check.
        """
        # Test classifier network
        
        # tanh multi-label classifier
        model = NeuralNetwork(objective=BinaryCrossEntropy)
        model.add(InputLayer(20, name='layer0'))
        model.add(DenseLayer(30, name='layer1', activation=Tanh))
        model.add(DenseLayer(20, name='layer2', activation=Sigmoid))
        model.build()
        assert model.grad_check(np.random.randn(10, 20), 
                                np.random.rand(10, 20))
                                
        # logistic multi-class classifier with regularization               
        model = NeuralNetwork(objective=CrossEntropy)
        model.add(InputLayer(20, name='layer0'))
        model.add(DenseLayer(30, name='layer1', 
                             activation=Sigmoid, 
                             W_regularizer=L2(0.1), 
                             b_regularizer=L2(0.1)))
        model.add(DenseLayer(20, name='layer2', 
                             activation=Softmax, 
                             W_regularizer=L2(0.01), 
                             b_regularizer=L2(0.01)))
        model.build()
        assert model.grad_check(np.random.randn(10, 20), 
                                utils.softmax(np.random.randn(10, 20)))
                                
        # deep multi-class classifier
        model = NeuralNetwork(objective=CrossEntropy)
        model.add(InputLayer(30, name='layer0'))
        model.add(DenseLayer(51, name='layer1', activation=Sigmoid))
        model.add(DenseLayer(50, name='layer2', 
                             activation=Tanh, 
                             W_regularizer=L2(0.1)))
        model.add(DenseLayer(49, name='layer3', 
                             activation=ReLu, 
                             W_regularizer=L2(0.1)))
        model.add(DenseLayer(52, name='layer4', 
                             activation=Identity))        
        model.add(DenseLayer(30, name='layer5', 
                             activation=Softmax))
        model.build()
        assert model.grad_check(np.random.randn(10, 30), 
                                utils.softmax(np.random.randn(10, 30)))
                                
        # Test regression network
        # single-output regression
        model = NeuralNetwork(objective=MeanSquareError)
        model.add(InputLayer(20, name='layer0'))
        model.add(DenseLayer(30, name='layer1', activation=ReLu))
        model.add(DenseLayer(30, name='layer2', activation=ReLu))
        model.add(DenseLayer(1, name='layer3', activation=ReLu))
        model.build()
        assert model.grad_check(np.random.randn(10, 20), 
                                np.random.randn(10, 1))
        
        # multi-output regression                                
        model = NeuralNetwork(objective=MeanSquareError)
        model.add(InputLayer(20, name='layer0'))
        model.add(DenseLayer(30, name='layer1', 
                             activation=Tanh, 
                             W_regularizer=L2(0.1)))        
        model.add(DenseLayer(30, name='layer2', 
                             activation=Tanh, 
                             W_regularizer=L2(0.1)))
        model.add(DenseLayer(10, name='layer3', 
                             activation=Identity, 
                             W_regularizer=L2(0.1)))
        model.build()
        assert model.grad_check(np.random.randn(10, 20), 
                                np.random.randn(10, 10))
        
    def test_fit(self):
        """Sanity check with small amount of random date.
        """
        
        model = NeuralNetwork(objective=CrossEntropy)
        model.add(InputLayer(30, name='layer0'))
        model.add(DenseLayer(100, name='layer1', activation=Tanh))
        model.add(DenseLayer(100, name='layer2', activation=Tanh))
        model.add(DenseLayer(2, name='layer4', activation=Softmax))
        model.build()
        
        x = np.random.randn(50, 30)
        y = utils.make_onehots(np.random.randint(0, 2, 50), (50, 2))
        cb = [(callbacks.IterationPrinter, 100),]
              
        model.fit(x, y, 
                  n_epochs=2000,
                  batch_size=32,
                  callbacks=cb, 
                  optimizer=SgdOptimizer(learning_rate=0.3))
        assert model.compute_loss(x, y) < 1e-4