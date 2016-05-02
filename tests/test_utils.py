# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:46:52 2016

@author: lifu
"""

import numpy as np

from dimsum.utils import ArrayPool

eps = 1e-9

class TestArrayPool:
    """Test class for ArrayPool.
    """
    def test_attribute_access(self):
        """Test attribute-style access to arrays.
        """

        a = np.random.randn(300, 400)
        b = np.random.randn(50, 60, 70)
        c = np.random.randn(1000)
        d = np.random.randn(200, 300)
        
        ap = ArrayPool({'a': a.shape, 'b': b.shape, 
                        'c': c.shape, 'd': d.shape})
            
        ap.a[:] = a
        ap.b[:] = b
        ap.c[:] = c
        ap.d[:] = d
        
        assert np.linalg.norm(ap.a - a) < eps
        assert np.linalg.norm(ap.b - b) < eps
        assert np.linalg.norm(ap.c - c) < eps
        assert np.linalg.norm(ap.d - d) < eps
        
    def test_attribute_redirection(self):
        """Test redirection of access to its underlying ndarray.
        """
        
        ap = ArrayPool({'a': (100, 200), 'b': (10, 10, 10, 10), 'c': (10,)})
        ap._vec[:] = np.random.randn(*ap._vec.shape)
        
        # test __len__
        assert len(ap) == len(ap._vec)
        
        # test __iter__
        assert all(ap._vec[i] == v for i, v in enumerate(ap))
        
        # test __getitem__
        assert all(ap[i] == ap._vec[i] for i in xrange(len(ap)))
        assert np.all(ap[0:50:2] == ap._vec[0:50:2])
        
        # test __setitem__
        lst = [1, 10, 20, 30, 40]
        ap[lst] = 10
        assert all(ap[i] == 10 for i in lst)
        ap[:] = 0
        assert not np.any(ap._vec)     
        ap[:] = 1
        assert np.all(ap[:] == 1)