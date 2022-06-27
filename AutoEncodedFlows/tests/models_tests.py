# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:05:31 2022

@author: William
"""

import torch
import unittest
from AutoEncodedFlows.models import Projection1D, SequentialFlow
from AutoEncodedFlows.models import CNFAutoEncoderFlowSCurve


class Projection1D_Tests(unittest.TestCase):
    
    def test_dimensionality(self):
        
        proj_layer = Projection1D(2,3)
        self.assertTrue(proj_layer.projection.shape == (2,3))
        
        proj_layer = Projection1D(3,2)
        self.assertTrue(proj_layer.projection.shape == (3,2))       
        
        
class SequentialFlow_Tests(unittest.TestCase):
    
    
    def test_sequentialflow(self):
        
        testflow = SequentialFlow(Projection1D(2,3),
                                  Projection1D(3,4))
        
        input_data = torch.ones((2,2))        
        output_data = testflow(input_data)
        inverse_data = testflow.inverse(output_data)
        
        self.assertTrue(output_data.shape == torch.Size([2,4]))
        self.assertTrue((inverse_data == input_data).all())
        
class CNFAutoEncoderFlowSCurve_Tests(unittest.TestCase):
    
    def test_encode_decode(self):
        
        model = CNFAutoEncoderFlowSCurve()
        test_data = torch.randn((10,3))
        
        enc_data = model.encode(test_data)
        dec_data = model.decode(enc_data)
        

                
if __name__ == '__main__':
    
    import os
    os.chdir(b"""C:\Users\William\Documents\UCL\Modules\MSc Thesis\MscThesis""")
        
    unittest.main()
    
    