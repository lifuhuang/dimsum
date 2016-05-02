# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:09:14 2016

@author: lifu
"""

import sys

import matplotlib.pyplot as plt

class LossPrinter(object):
    """Output loss to console.
    """
    
    
    def __init__(self, x=None, y=None, outfd=sys.stderr):
        self._x = x
        self._y = y
        self._outfd = outfd
    
    def __call__(self, msg):
        model = msg['model']
        if self._x is None and self._y is None:
            x, y = msg['x_batch'], msg['y_batch']
        else:
            x, y = self._x, self._y
            
        loss = model.compute_loss(x, y)
        print >> self._outfd, 'Loss on dataset: %.08f' % loss
    
class LossPlotter(object):
    """Plot loss curves.
    """
    
    
    def __init__(self, outfd=sys.stderr):
        plt.ion()
        plt.axis([0, 500, 0, 0.1])
        plt.show()
        
    def __call__(self, msg):
        n_iters = msg['optimizer'].n_iters
        #adjust axis
        while n_iters * 1.2 > plt.xlim()[1]:
                plt.xlim(xmax = n_iters * 1.2)
        if state.cost >= plt.ylim()[1]:
            plt.ylim(ymax = state.cost * 1.05)
        # plot min/max point
        if state.cost < min_cost:
            plt.plot(state.it, state.cost, 'go')
            min_cost = state.cost
        elif state.cost > max_cost:                    
            plt.plot(state.it, state.cost, 'ro')
            max_cost = state.cost
        # plot title
        plt.title('cost: %g, min: %g' % (state.cost, min_cost))                
        # plot lines
        if old_cost:
            line = 'r-' if state.cost > old_cost else 'g-'
            plt.plot([state.it - args.display, state.it], 
                     [old_cost, state.cost], line)
        plt.draw()
        plt.pause(0.1)
        # record old cost
        old_cost = state.cost

class IterationPrinter(object):
    """Output iteration info to console.
    """
    
    
    def __init__(self, outfd=sys.stderr):
        self._outfd = outfd
    
    def __call__(self, msg):
        print >> self._outfd, 'Iteration %d' % msg['optimizer'].n_iters
    