# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:09:14 2016

@author: lifu
"""

import sys

class LossPrinter(object):
    """Output loss to console.
    """
    
    
    def __init__(self, x=None, y=None, outfd=sys.stderr, update_rate=1.0):
        self._x = x
        self._y = y
        self._outfd = outfd
        self._update_rate = update_rate
        self._loss = None
        
    def __call__(self, msg):
        model = msg['model']
        if self._x is None and self._y is None:
            x, y = msg['x_batch'], msg['y_batch']
        else:
            x, y = self._x, self._y
            
        if self._loss is None:
            self._loss = model.compute_loss(x, y)
        else:
            self._loss = (model.compute_loss(x, y) * self._update_rate + 
                        self._loss * (1.0 - self._update_rate))
                        
        print >> self._outfd, 'Loss on dataset: %.08f' % self._loss

class IterationPrinter(object):
    """Output iteration info to console.
    """
    
    
    def __init__(self, outfd=sys.stderr):
        self._outfd = outfd
    
    def __call__(self, msg):
        print >> self._outfd, 'Iteration %d' % msg['n_iters']
    