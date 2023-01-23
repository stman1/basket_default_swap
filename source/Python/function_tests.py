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

from scipy.special import gamma

# Unit tests for functions

print(f'***** TEST 1 ***** TEST INTEREST RATE CURVE PARSER ***** parse_interest_rate_curve *****')

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

# TEST PSEUDO-SAMPLE CSV PARSER

print(f'***** TEST 2 ***** TEST PSEUDO-SAMPLE CSV PARSER ***** parse_pseudo_samples *****')

from functions import parse_pseudo_samples

data_set_directory_name = 'PseudoSamples'
data_set_file_name = 'pseudo.samples.csv'
data_set_headers =  ['Prudential','BMW','Volkswagen','Deutsche Bank', 'Kering']

pseudo_sample_df = parse_pseudo_samples(data_set_directory_name, data_set_file_name, data_set_headers)

print(pseudo_sample_df.head())


# TEST OF LOG-LINEAR INTERPOLATION OF DISCOUNT FACTORS
 
print(f'***** TEST 3 ***** TEST OF LOG-LINEAR INTERPOLATION OF DISCOUNT FACTORS ***** loglinear_discount_factor *****')
from functions import loglinear_discount_factor
maturity = [0.5/12, 0.75/12, 1/12, 1/6, 1/4, 1/3, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 15/12, 0.75, 21/12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30]
t = np.arange(29)
cds_discount_factors = np.zeros(len(t))
 
for i in range(0, len(t)):
    cds_discount_factors[i] = loglinear_discount_factor(maturity, ir_data_frame['Discount Factor ACT360'], t[i])
     
print(f'discount factors: {cds_discount_factors}')


# TEST STUDENT T DENSITY CALCULATOR
print(f'***** TEST 4 ***** TEST STUDENT-T DENSITY CALCULATOR ***** student_t_copula_density *****')
from functions import student_t_copula_density

# input parameters
n = 5
uniform_pseudo_sample = np.random.uniform(0, 1, n)
nu = 1

# create a fake correlation 
sigma_independent = np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])

sigma_regular_dependence = np.array([[1., 0.2, 0.2, 0.2, 0.2],
       [0.2, 1., 0.2, 0.2, 0.2],
       [0.2, 0.2, 1., 0.2, 0.2],
       [0.2, 0.2, 0.2, 1., 0.2],
       [0.2, 0.2, 0.2, 0.2, 1.]])

sigma_irregular_dependence = np.array([[1., 0.8, 0.6, 0.4, 0.2],
       [0.8, 1., 0.8, 0.6, 0.4],
       [0.6, 0.8, 1., 0.8, 0.6],
       [0.4, 0.6, 0.8, 1., 0.8],
       [0.2, 0.4, 0.6, 0.8, 1.]])

# Assign sigma

sigma = sigma_independent

sigma_det = np.linalg.det(sigma)
sigma_inverse = np.linalg.inv(sigma)

# compute all components of the Student-t density formula that do not depend on the pseudo sample
f1 = 1. / np.sqrt(sigma_det)
f2 = gamma((nu + n )/ 2.) / gamma(nu / 2.)
f3 = np.power(gamma(nu / 2.) / gamma((nu + 1.) / 2.), n)

density = student_t_copula_density(uniform_pseudo_sample, n, nu, f1, f2, f3, sigma_det, sigma_inverse)

print(f'probability density fo parameter nu = {nu}: {density}')

# TEST MAXIMUM LIKELIHOOD ESTIMATION OF DEGREE OF FREEDOM PARAMETER FOR STUDENT T 
print(f'***** TEST 5 ***** TEST STUDENT-T LOG LIKELIHOOD COMPUTATION FOR GIVEN DEGREE OF FREEDOM PARAMETER \nu  ***** student_t_loglikelihood *****')
from functions import student_t_loglikelihood

nu = 1
sigma = sigma_independent

loglikelihood = student_t_loglikelihood(pseudo_sample_df, nu, sigma)

print(f'loglikelihood for dof parameter nu = {nu}: {loglikelihood}')

# TEST MAXIMUM LIKELIHOOD ESTIMATION OF DEGREE OF FREEDOM PARAMETER FOR STUDENT T 
print(f'***** TEST 6 ***** TEST MAXIMUM LIKELIHOOD ESTIMATION OF DEGREE OF FREEDOM PARAMETER FOR STUDENT T *****  *****')
from functions import maximum_likelihood_student_t_dof


sigma_independent = np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])

sigma_regular_dependence = np.array([[1., 0.2, 0.2, 0.2, 0.2],
       [0.2, 1., 0.2, 0.2, 0.2],
       [0.2, 0.2, 1., 0.2, 0.2],
       [0.2, 0.2, 0.2, 1., 0.2],
       [0.2, 0.2, 0.2, 0.2, 1.]])

sigma_irregular_dependence = np.array([[1., 0.8, 0.6, 0.4, 0.2],
       [0.8, 1., 0.8, 0.6, 0.4],
       [0.6, 0.8, 1., 0.8, 0.6],
       [0.4, 0.6, 0.8, 1., 0.8],
       [0.2, 0.4, 0.6, 0.8, 1.]])

sigma = sigma_irregular_dependence

maximum_likelihood_dict = maximum_likelihood_student_t_dof(pseudo_sample_df, sigma, plot_likelihood=False)

max_likelihood_nu = max(maximum_likelihood_dict, key=lambda key : maximum_likelihood_dict[key])

print(f'maximum likelihood for dof parameter nu = {max_likelihood_nu}')




