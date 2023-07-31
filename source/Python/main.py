# -*- coding: utf-8 -*-
import os
import scipy as sc
import numpy as np
import pandas as pd
from scipy.stats import qmc
from n_th_to_default import parse_yield_curve_from_excel
from functions import parse_interest_rate_curve, loglinear_discount_factor


"""
Created on Thu Jan 19 20:46:03 2023

@author: Stefan Mangold
"""

# Gaussian copula k-th to default credit basket

# preliminaries

# discount curve 
work_dir = os.getcwd()
cur_dir = os.chdir('../..')  # move two directories up
data_set = 'final_basket'
data_set_directory = os.chdir("%s%s%s"%('data','/', data_set)) 
ir_data_frame = parse_interest_rate_curve(data_set_directory, 
                                           'CDS_spreads_basket.xlsx', 
                                           'ESTR', 
                                           4, 
                                           'B:E', 
                                           ['Instr.Name', 'Close','START DATE','Mat.Dat'])

# bootstrap implied default probabilities




# pseudo samples

work_dir = os.getcwd()
cur_dir = os.chdir('../..')
data_set = 'PseudoSamples'
data_set_directory = os.chdir("%s%s%s"%('data','/', data_set)) 

pseudo_sample_data_frame = pd.read_csv('pseudo.samples.csv', 
                                       header=0, 
                                       names= ['Prudential','BMW','Volkswagen','Deutsche Bank', 'Kering'])  

# compute correlation matrix Gaussian copula

correlation_matrix = pseudo_sample_data_frame.corr().to_numpy()

#   check positive-definiteness property of estimated correlation matrix
print(f'All eigenvalues of correlation matrix are positive: {np.linalg.eig(correlation_matrix)[0].all() > 0}')
