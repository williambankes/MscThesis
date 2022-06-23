# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:05:31 2022

@author: William
"""

import unittest
from AutoEncodedFlows.models import Projection1D


class Projection1D_Tests(unittest.TestCase):
    
    def test_dimensionality(self):
        
        proj_layer = Projection1D(2,3)
        self.assertTrue(proj_layer.projection.shape == (2,3))
        
        proj_layer = Projection1D(3,2)
        self.assertTrue(proj_layer.projection.shape == (3,2))       
        
        
if __name__ == '__main__':
        
    unittest.main()
    
    