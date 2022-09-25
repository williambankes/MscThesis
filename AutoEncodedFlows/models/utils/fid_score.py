# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 12:51:36 2022

@author: William
"""

import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import transforms


class fid_scorer:
    
    def __init__(self):
        """
        Implementation of the Frechet Inception Distance utilising the pre-trained
        Inception_v3 model from pytorch and code taken from:
            
        https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

        Constructor loads inception model locally via <torch.hub.load>        

        Returns
        -------
        None.

        """
        
        #init model:
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'inception_v3', pretrained=True)
        
        #init transforms:
        self.data_transforms =  transforms.Compose([
                                transforms.Resize(299),
                                transforms.CenterCrop(299),
                                transforms.Lambda(lambda x: x.repeat(1,3, 1, 1)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.dev)

    def calculate_fid(self, act1, act2):
        """
        Implementation of Frechet Inception Distance given two sets of activation
        functions from the Inception V3 model

        Parameters
        ----------
        act1 : torch.tensor
            Activation layer of either real or generated samples
        act2 : TYPE
            Activation layer of either real or generated samples

        Returns
        -------
        fid : float
            Frechet Inception Distance between the two image batches
        """
        
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # calculate score        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


    def score(self, x_pred, x_real):
        """
        Calculates the FID score given two batches of images

        Parameters
        ----------
        x_pred : torch.tensor (N, C, H, W)
            Generated images from model
        x_real : torch.tensor (N, C, H, W)
            Real sample images

        Returns
        -------
        fid : <float>
            Frechet Inception Distance between the two input batches
        """
        x_pred, x_real = x_pred.cpu(), x_real.cpu()
        x_pred, x_real = self.data_transforms(x_pred), self.data_transforms(x_real)
        
        #create the model feature rep:
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.model.avgpool.register_forward_hook(get_activation("avgpool"))
        
        _ = self.model(x_pred)
        x_pred_act = activation['avgpool'].reshape(-1, 2048).cpu().detach().numpy()
        
        _ = self.model(x_real)
        x_real_act = activation['avgpool'].reshape(-1, 2048).cpu().detach().numpy()
            
        return self.calculate_fid(x_pred_act, x_real_act)    

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from torchvision.datasets import FashionMNIST
    
    scorer = fid_scorer()
    data = FashionMNIST('../../', download=True, train=True,
                        transform=transforms.ToTensor())
    
    real_sample = torch.stack([d[0] for d in data][:10])
    
    output = scorer.score(real_sample, real_sample)
    output2 = scorer.score(real_sample + torch.randn([10,1,28,28]), real_sample)
    
    print('real vs real', output)
    print('real vs noisy', output2)
    
    noise_image = real_sample[0] + torch.randn([1,28,28])
    plt.imshow(noise_image.reshape(28,28,1).numpy())