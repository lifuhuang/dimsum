# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:27:21 2016

@author: lifu
"""

import numpy as np

from dimsum.layers import DenseLayer, InputLayer
from dimsum.models import NeuralNetwork
from dimsum import utils
from dimsum import callbacks

class TestNeuralNetwork:
    """Test class for NeuralNetwork.
    """
        
    def test_fit(self):
        """Sanity check with small amount of random date.
        """
        
        model = NeuralNetwork()
        model.add(InputLayer(30, name='layer0'))
        model.add(DenseLayer(100, name='layer1', activation='tanh'))
        model.add(DenseLayer(100, name='layer2', activation='tanh'))
        model.add(DenseLayer(2, name='layer4', activation='softmax'))
        model.build()
        
        x = np.random.randn(50, 30)
        y = utils.make_onehots(np.random.randint(0, 2, 50), (50, 2))
        optimizer_args={'learning_rate': 0.3}
        cb = [(callbacks.print_iteration_info, 100), 
              (lambda m, r: callbacks.print_loss(m, r, x, y), 100)]
              
        sample_iter = utils.random_iter(x, y, n_epochs=1000)
        model.fit(sample_iter, cb, optimizer_args)
        assert model.compute_loss(x, y) < 1e-4
    
    def test_grad_check(self):
        """Perform gradient check.
        """
        # Test classifier network
        # logistic softmax classifier                       
        model = NeuralNetwork()
        model.add(InputLayer(20, name='layer0'))
        model.add(DenseLayer(30, name='layer1', activation='logistic'))
        model.add(DenseLayer(20, name='layer2', activation='softmax'))
        model.build()
        assert model.grad_check(np.random.randn(10, 20), 
                                utils.softmax(np.random.randn(10, 20)))
                                
        # deep softmax classifier
        model = NeuralNetwork()
        model.add(InputLayer(30, name='layer0'))
        model.add(DenseLayer(51, name='layer1', activation='logistic'))
        model.add(DenseLayer(50, name='layer2', activation='tanh'))
        model.add(DenseLayer(49, name='layer3', activation='relu'))
        model.add(DenseLayer(52, name='layer4', activation='identity'))        
        model.add(DenseLayer(30, name='layer5', activation='softmax'))
        model.build()
        assert model.grad_check(np.random.randn(10, 30), 
                                utils.softmax(np.random.randn(10, 30)))
        # tanh sigmoid classifier
        model = NeuralNetwork()
        model.add(InputLayer(20, name='layer0'))
        model.add(DenseLayer(30, name='layer1', activation='tanh'))
        model.add(DenseLayer(20, name='layer2', activation='logistic'))
        model.build()
        assert model.grad_check(np.random.randn(10, 20), 
                                utils.softmax(np.random.randn(10, 20)))
                                
        # Test regression network
        # single-output regression
        model = NeuralNetwork(objective='mean_square')
        model.add(InputLayer(20, name='layer0'))
        model.add(DenseLayer(30, name='layer1', activation='relu'))        
        model.add(DenseLayer(30, name='layer2', activation='relu'))
        model.add(DenseLayer(1, name='layer3', activation='relu'))
        model.build()
        assert model.grad_check(np.random.randn(10, 20), 
                                np.random.randn(10, 1))
        
        # multi-output regression                                
        model = NeuralNetwork(objective='mean_square')
        model.add(InputLayer(20, name='layer0'))
        model.add(DenseLayer(30, name='layer1', activation='tanh'))        
        model.add(DenseLayer(30, name='layer2', activation='tanh'))
        model.add(DenseLayer(10, name='layer3', activation='identity'))
        model.build()
        assert model.grad_check(np.random.randn(10, 20), 
                                np.random.randn(10, 10))
    