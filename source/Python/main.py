# -*- coding: utf-8 -*-
import scipy as sc
import numpy as np
from scipy.stats import qmc


"""
Created on Tue Jul  5 07:29:03 2022

@author: Stefan Mangold
"""


def sobol_sequence():
    pass

def multivariate_random_normal(mu, sigma, num_simulations):
    
    
    
    pass



sampler = qmc.Sobol(d=5, scramble=False)
sample = sampler.random_base2(m=7)
print(sample)

print(qmc.discrepancy(sample))


#rng = np.random.default_rng(12345)
#print(rng)

#rfloat = rng.random(size=128)
#print(rfloat)