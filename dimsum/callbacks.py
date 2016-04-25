# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:09:14 2016

@author: lifu
"""
import sys

def print_loss(model, optimizer, x, y, fmt=None, outfd=sys.stderr):
    """Output loss to console.
    """
    
    loss = model.compute_loss(x, y)
    if fmt is None:
        fmt = 'Loss on dataset: %.08f'
    print(fmt % loss)
    
def print_iteration_info(model, optimizer):
    """Output iteration info to console.
    """
    
    msg = 'Iteration %d' % optimizer.n_iters
    print(msg)