# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:27:21 2016

@author: lifu
"""

import numpy as np

from dimsum.layers import DenseLayer, InputLayer
from dimsum.nn import NeuralNetwork
from dimsum import utils

class TestNeuralNetwork:
    """Test class for NeuralNetwork.
    """
    
    def test_grad_check(self):
        """Perform gradient check.
        """
        
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
