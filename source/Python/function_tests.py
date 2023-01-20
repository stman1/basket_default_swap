#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 05:45:17 2023

Unit tests for functions defined in functions.py

@author: Stefan Mangold
"""
import os
import numpy as np
import pandas as pd

# Unit tests for functions

# TEST INTEREST RATE CURVE PARSER

from functions import parse_interest_rate_curve
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

print(f'raw interest rates: {ir_data_frame}')

# Test linearization of rank correlation matrix
from functions import linearize_spearman_correlation_matrix
square_matrix_of_threes = np.ones([5,5]) * 3
linearized_ones = linearize_spearman_correlation_matrix(square_matrix_of_threes)


# TEST OF LOG-LINEAR INTERPOLATION OF DISCOUNT FACTORS
 
from functions import loglinear_discount_factor
maturity = [0.5/12, 0.75/12, 1/12, 1/6, 1/4, 1/3, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 15/12, 0.75, 21/12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30]
t = np.arange(29)
cds_discount_factors = np.zeros(len(t))
 
for i in range(0, len(t)):
    cds_discount_factors[i] = loglinear_discount_factor(maturity, ir_data_frame['Discount Factor ACT360'], t[i])
     
print(f'discount factors: {cds_discount_factors}')


# TEST STUDENT T DENSITY CALCULATOR

from functions import student_t_copula_density

n = 5
uniform_pseudo_sample = np.random.uniform(0, 1, n)
nu = 1


density = student_t_copula_density(uniform_pseudo_sample, n, nu, sigma)

