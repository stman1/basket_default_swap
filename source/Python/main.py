# -*- coding: utf-8 -*-
import os
import scipy as sc
import numpy as np
from scipy.stats import qmc
from n_th_to_default import parse_yield_curve_from_excel


"""
Created on Tue Jul  5 07:29:03 2022

@author: Stefan Mangold
"""


def sobol_sequence():
    pass

def multivariate_random_normal(mu, sigma, num_simulations):
    pass

#sampler = qmc.Sobol(d=5, scramble=False)
#sample = sampler.random_base2(m=7)
#print(sample)

#print(qmc.discrepancy(sample))


#rng = np.random.default_rng(12345)
#print(rng)

#rfloat = rng.random(size=128)
#print(rfloat)


os.chdir('..\..') # move two directories up
path = 'data\default_basket_data.xlsx'
tab = 'ESTR'

yield_curve_frame = parse_yield_curve_from_excel(path, tab)
    
    
