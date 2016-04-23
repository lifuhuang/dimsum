# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:09:59 2016

@author: lifu
"""

import numpy as np

def cross_entropy_loss(output, target):
    return -np.dot(target.T, np.log(output))

def cross_entropy_derivative(output, target):
    return -target / output
    
functions = {'cross_entropy': cross_entropy_loss}
derivatives = {'cross_entropy': cross_entropy_derivative}