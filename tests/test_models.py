# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:27:21 2016

@author: lifu
"""

import numpy as np

from dimsum.layers import Dense, Input
from dimsum.optimizers import SGD
from dimsum.regularizers import L2
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
        model = NeuralNetwork(objective='bcee')
        model.add(Input(20))
        model.add(Dense(20, 30, activation='tanh'))
        model.add(Dense(30, 20, activation='sigmoid'))
        assert model.grad_check(np.random.randn(10, 20), 
                                np.random.rand(10, 20))
                                
        # logistic multi-class classifier with regularization               
        model = NeuralNetwork(objective='cee')
        model.add(Input(20))
        model.add(Dense(20, 30, activation='sigmoid', W_regularizer=L2(0.1), 
                        b_regularizer=L2(0.1)))
        model.add(Dense(30, 20, activation='softmax', W_regularizer=L2(0.01), 
                        b_regularizer=L2(0.01)))
        assert model.grad_check(np.random.randn(10, 20), 
                                utils.softmax(np.random.randn(10, 20)))
                                
        # deep multi-class classifier
        model = NeuralNetwork(objective='cee')
        model.add(Input(30))
        model.add(Dense(30, 51, activation='sigmoid'))
        model.add(Dense(51, 50, activation='tanh', W_regularizer=L2(0.1)))
        model.add(Dense(50, 49, activation='relu', W_regularizer=L2(0.1)))
        model.add(Dense(49, 52, activation='identity'))
        model.add(Dense(52, 30, activation='softmax'))
        assert model.grad_check(np.random.randn(10, 30), 
                                utils.softmax(np.random.randn(10, 30)))
                                
        # Test regression network
        # single-output regression
        model = NeuralNetwork(objective='mse')
        model.add(Input(20))
        model.add(Dense(20, 30, activation='relu'))
        model.add(Dense(30, 30, activation='relu'))
        model.add(Dense(30, 1, activation='relu'))
        assert model.grad_check(np.random.randn(10, 20), 
                                np.random.randn(10, 1))
        
        # multi-output regression                                
        model = NeuralNetwork(objective='mse')
        model.add(Input(20))
        model.add(Dense(20, 30, activation='tanh', W_regularizer=L2(0.1)))        
        model.add(Dense(30, 30, activation='tanh', W_regularizer=L2(0.1)))
        model.add(Dense(30, 10, activation='identity', W_regularizer=L2(0.1)))
        assert model.grad_check(np.random.randn(10, 20), 
                                np.random.randn(10, 10))
        
    def test_fit(self):
        """Sanity check with small amount of random date.
        """
        

        
        model = NeuralNetwork(objective='cee')
        model.add(Input(30))
        model.add(Dense(30, 100, activation='tanh'))
        model.add(Dense(100, 100, activation='tanh'))
        model.add(Dense(100, 2, activation='softmax'))
        
        x = np.random.randn(50, 30)
        y = utils.make_onehots(np.random.randint(0, 2, 50), (50, 2))
        cb = [(callbacks.IterationPrinter, 100),]
              
        model.fit(x, y, 
                  n_epochs=2000,
                  batch_size=32,
                  callbacks=cb, 
                  optimizer=SGD(learning_rate=0.3))
        assert model.compute_loss(x, y) < 1e-4