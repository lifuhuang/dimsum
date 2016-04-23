# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:27:21 2016

@author: lifu
"""



if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add(DenseLayer(10, 50, name='layer1', nonlinearity='logistic'))
    nn.add(DenseLayer(50, 30, name='layer2', nonlinearity='tanh'))
    nn.add(DenseLayer(30, 20, name='layer3', nonlinearity='softmax'))
    nn.add(DenseLayer(20, 30, name='layer4', nonlinearity='identity'))    
    nn.add(DenseLayer(30, 30, name='layer5', nonlinearity='softmax'))
    nn.build()
    nn.grad_check(np.random.randn(10, 10), nonlinearities.softmax(np.random.randn(10, 30)))