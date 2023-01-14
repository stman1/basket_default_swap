# -*- coding: utf-8 -*-
import os
import scipy as sc
import numpy as np
from scipy.stats import qmc
from n_th_to_default import parse_yield_curve_from_excel
from functions import parse_interest_rate_curve, loglinear_discount_factor


"""
Created on Tue Jul  5 07:29:03 2022

@author: Stefan Mangold
"""

# LOAD INTEREST RATE INFORMATION AND COMPUTE DISCOUNT FACTORS

work_dir = os.getcwd()
cur_dir = os.chdir('../..')  # move two directories up
data_set = 'Banks'
data_set_directory = os.chdir("%s%s%s"%('data','/', data_set)) 
ir_data_frame = parse_interest_rate_curve(data_set_directory, 
                                          'CDS_spreads.xlsx', 
                                          'ESTR', 
                                          2, 
                                          'B:E', 
                                          ['Instr.Name', 'Close','START DATE','Mat.Dat'])


# TEST LOG-LINEAR INTERPOLATION OF DISCOUNT FACTORS

maturity = [0.5/12, 0.75/12, 1/12, 1/6, 1/4, 1/3, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 15/12, 0.75, 21/12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30
            ]
t = np.arange(29)
cds_discount_factors = np.zeros(len(t))

for i in range(0, len(t)):
    cds_discount_factors[i] = loglinear_discount_factor(maturity, ir_data_frame['Discount Factor ACT360'], t[i])
    
print(f'discount factors: {cds_discount_factors}')



