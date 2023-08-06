# -*- coding: utf-8 -*-
import os
import scipy as sc
import numpy as np
import pandas as pd
from scipy.stats import qmc
from n_th_to_default import parse_yield_curve_from_excel

from functions import parse_interest_rate_curve, loglinear_discount_factor
from functions import cds_bootstrapper, loglinear_discount_factor


"""
Created on Thu Jan 19 20:46:03 2023

@author: Stefan Mangold
"""

# Spread calculation of a k-th to default credit basket swap with Gaussian copula

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

recovery = 0.4

t = [1, 2, 3, 4, 5]

# need cds spread parser
t = [1, 2, 3, 4, 5]
t_new = [0, 0.125, 0.375, 0.55, 0.9, 5, 5.5]
prud = [29.5, 40.13, 50.6, 63.16, 74.18]
bmw = [28, 37.92, 47.09, 58.16, 70.2]
vw = [69.29, 81.66, 97, 111.93, 131.64]
db = [85.5, 91.32, 97, 103.49, 111.45]
ker = [13.28, 18.65, 24.05, 31.15, 38.24]
maturity = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.]
discount_factor = [0.9925,	0.9851,	0.9778,	0.9704,	0.9632,	0.9560,	0.9489,	0.9418,	0.9347,	0.9277,	0.9208,	0.9139,	0.9071,	0.9003,	0.8936,	0.8869,	0.8803,	0.8737,	0.8672,	0.8607]
df = loglinear_discount_factor(maturity, discount_factor, t)
cds_df = pd.DataFrame({'Maturity' : t, 'prudential_spreads' : prud, 'bmw_spreads' : bmw, 'volkswagen_spreads' : vw, 'deutsche_bank_spreads' : db, 'kering_spreads' : ker, 'discount_factor': df})


spreads_prudential = cds_bootstrapper(cds_df.Maturity, cds_df.discount_factor, cds_df.prudential_spreads, recovery, plot_prob=False, plot_hazard=False)
spreads_bmw = cds_bootstrapper(cds_df.Maturity, cds_df.discount_factor, cds_df.bmw_spreads, recovery, plot_prob=False, plot_hazard=False)
spreads_vw = cds_bootstrapper(cds_df.Maturity, cds_df.discount_factor, cds_df.volkswagen_spreads, recovery, plot_prob=False, plot_hazard=False)
spreads_db = cds_bootstrapper(cds_df.Maturity, cds_df.discount_factor, cds_df.deutsche_bank_spreads, recovery, plot_prob=False, plot_hazard=False)
spreads_ker = cds_bootstrapper(cds_df.Maturity, cds_df.discount_factor, cds_df.kering_spreads, recovery, plot_prob=False, plot_hazard=False)




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
