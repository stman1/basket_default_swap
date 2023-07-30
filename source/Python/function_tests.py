#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 05:45:17 2023

Unit tests for functions defined in file functions.py

@author: Stefan Mangold
"""
import os
import numpy as np
import pandas as pd

from scipy.special import gamma

# Unit tests for functions

# MOCK TEST OBJECTS 

# create mock correlation matrices
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



# =============================================================================
# print(f'***** TEST 1 ***** TEST INTEREST RATE CURVE PARSER ***** parse_interest_rate_curve *****')
# 
# from functions import parse_interest_rate_curve
# work_dir = os.getcwd()
# cur_dir = os.chdir('../..')  # move two directories up
# data_set = 'final_basket'
# data_set_directory = os.chdir("%s%s%s"%('data','/', data_set)) 
# ir_data_frame = parse_interest_rate_curve(data_set_directory, 
#                                            'CDS_spreads_basket.xlsx', 
#                                            'ESTR', 
#                                            4, 
#                                            'B:E', 
#                                            ['Instr.Name', 'Close','START DATE','Mat.Dat'])
# 
# print(f'raw interest rates: {ir_data_frame}')
# 
# =============================================================================
# Test linearisation of rank correlation matrix
# print(f'***** TEST 2 ***** LINEARISATION OF SPEARMAN RANK CORRELATION MATRIX ***** linearise_spearman_correlation_matrix *****')
# from functions import linearise_spearman_correlation_matrix
# square_matrix_of_threes = np.ones([5, 5]) * 3
# linearised_correlation_matrix = linearise_spearman_correlation_matrix(square_matrix_of_threes)
# print(f'Linearised matrix: {linearised_correlation_matrix}')
# =============================================================================
 
# # TEST PSEUDO-SAMPLE CSV PARSER
# 
# print(f'***** TEST 2 ***** TEST PSEUDO-SAMPLE CSV PARSER ***** parse_pseudo_samples *****')
# 
# from functions import parse_pseudo_samples
# 
# data_set_directory_name = 'PseudoSamples'
# data_set_file_name = 'pseudo.samples.csv'
# data_set_headers =  ['Prudential','BMW','Volkswagen','Deutsche Bank', 'Kering']
# 
# pseudo_sample_df = parse_pseudo_samples(data_set_directory_name, data_set_file_name, data_set_headers)
# 
# print(pseudo_sample_df.head())
# 
# 
# TEST OF LOG-LINEAR INTERPOLATION OF DISCOUNT FACTORS
  
# =============================================================================
# print(f'***** TEST 3 ***** TEST OF LOG-LINEAR INTERPOLATION OF DISCOUNT FACTORS ***** loglinear_discount_factor *****')
# from functions import loglinear_discount_factor
# maturity = [0.5/12, 0.75/12, 1/12, 1/6, 1/4, 1/3, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 15/12, 0.75, 21/12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30]
# t = np.arange(29)
# cds_discount_factors = np.zeros(len(t))  
# for i in range(0, len(t)):
#     cds_discount_factors[i] = loglinear_discount_factor(maturity, ir_data_frame['Discount Factor ACT360'], t[i])
#       
# print(f'discount factors: {cds_discount_factors}')
#  
#  
# =============================================================================
# # TEST STUDENT T DENSITY CALCULATOR
# print(f'***** TEST 4 ***** TEST STUDENT-T DENSITY CALCULATOR ***** student_t_copula_density *****')
# from functions import student_t_copula_density
# 
# # input parameters
# n = 5
# uniform_pseudo_sample = np.random.uniform(0, 1, n)
# nu = 1
# 
# # Assign correlation matrix sigma
# sigma = sigma_independent
# 
# sigma_det = np.linalg.det(sigma)
# sigma_inverse = np.linalg.inv(sigma)
# 
# # compute all components of the Student-t density formula that do not depend on the pseudo sample
# f1 = 1. / np.sqrt(sigma_det)
# f2 = gamma((nu + n) / 2.) / gamma(nu / 2.)
# f3 = np.power(gamma(nu / 2.) / gamma((nu + 1.) / 2.), n)
# 
# density = student_t_copula_density(uniform_pseudo_sample, n, nu, f1, f2, f3, sigma_det, sigma_inverse)
# 
# print(f'probability density fo parameter nu = {nu}: {density}')
# 
# # TEST MAXIMUM LIKELIHOOD ESTIMATION OF DEGREE OF FREEDOM PARAMETER FOR STUDENT T 
# print(f'***** TEST 5 ***** TEST STUDENT-T LOG LIKELIHOOD COMPUTATION FOR GIVEN DEGREE OF FREEDOM PARAMETER \nu  ***** student_t_loglikelihood *****')
# from functions import student_t_loglikelihood
# 
# nu = 1
# sigma = sigma_independent
# 
# loglikelihood = student_t_loglikelihood(pseudo_sample_df, nu, sigma)
# 
# print(f'loglikelihood for dof parameter nu = {nu}: {loglikelihood}')
# 
# # TEST MAXIMUM LIKELIHOOD ESTIMATION OF DEGREE OF FREEDOM PARAMETER FOR STUDENT T 
# print(f'***** TEST 6 ***** TEST MAXIMUM LIKELIHOOD ESTIMATION OF DEGREE OF FREEDOM PARAMETER FOR STUDENT T *****  *****')
# from functions import maximum_likelihood_student_t_dof
# 
# sigma = sigma_regular_dependence
# 
# maximum_likelihood_dict = maximum_likelihood_student_t_dof(pseudo_sample_df, sigma, plot_likelihood=True)
# 
# max_likelihood_nu = max(maximum_likelihood_dict, key=lambda key : maximum_likelihood_dict[key])
# 
# print(f'maximum likelihood for dof parameter nu = {max_likelihood_nu}')
# 
# 
# # TEST SAMPLING FROM GAUSSIAN COPULA
# print(f'***** TEST 7 ***** TEST SAMPLING FROM GAUSSIAN COPULA *****  *****')
# from functions import sampling_gaussian_copula
# 
# # assign correlation matrix sigma
# correlation_matrix = sigma_regular_dependence
# 
# correlated_uniform_sample = sampling_gaussian_copula(correlation_matrix, dimension=5, power_of_two = 4)
# 
# print(f'Correlated uniform sample shape = {correlated_uniform_sample.shape}')
# =============================================================================

# TEST SAMPLING FROM STUDENT-T COPULA
# =============================================================================
# print(f'***** TEST 8 ***** TEST SAMPLING FROM STUDENT-T COPULA ***** sampling_student_t_copula *****')
# from functions import sampling_student_t_copula
# 
# # assign correlation matrix sigma
# correlation_matrix = sigma_regular_dependence
# 
# correlated_uniform_sample = sampling_student_t_copula(correlation_matrix, 7, dimension=5, power_of_two = 4)
# 
# print(f'Correlated uniform sample shape = {correlated_uniform_sample.shape}')
# =============================================================================

# =============================================================================
# # TEST PREMIUM LEG COMPUTATION
# # =============================================================================
# print('***** TEST 9 ***** TEST PREMIUM LEG COMPUTATION ***** premium_leg *****')
# from functions import parse_interest_rate_curve, calc_premium_leg
# 
# # general input arguments
# 
# expiry = 5 # in years
# payment_frequency = 4 # how often default is observed / payments are scheduled
# 
# # get interest rate curve
# 
# work_dir = os.getcwd()
# cur_dir = os.chdir('../..')  # move two directories up
# data_set = 'final_basket'
# data_set_directory = os.chdir("%s%s%s"%('data','/', data_set)) 
# interest_rate_curve = parse_interest_rate_curve(data_set_directory, 
#                                            'CDS_spreads_basket.xlsx', 
#                                            'TEST_CURVE', 
#                                            4, 
#                                            'B:E', 
#                                            ['Instr.Name', 'Close','START DATE','Mat.Dat'])
# 
# # Case 1: All defaults after expiry
# print('Case 1: All defaults after expiry')
# default_times = np.array([9, 7, 6, 5.5, 8]) # all default times > expiry, unsorted
# k = 1 # 1st to default, number of protected defaults before the cds expires is 1 
# pv_premium_leg = calc_premium_leg(expiry, default_times, payment_frequency, k, interest_rate_curve)
# print(f'PV Premium leg = pv_premium_leg = {pv_premium_leg}')
# 
# 
# # Case 2: defaults occur before expiry, all are protected (two defaults before expiry, k = 3)
# print(' Case 2: One or more defaults before expiry, all are protected (two defaults before expiry, k = 3)')
# default_times = np.array([9, 7, 3.2, 2.7, 1.45]) # at least one default occurs before expiry
# k = 3 # 1st to default 
# pv_premium_leg = calc_premium_leg(expiry, default_times, payment_frequency, k, interest_rate_curve)
# print(f'PV Premium leg = pv_premium_leg = {pv_premium_leg}')
# 
# 
# # Case 3: defaults occur before expiry and not all are protected
# print(' Case 3: defaults occur before expiry and not all are protected (two defaults before expiry, k = 2)')
# default_times = np.array([9, 7, 3.2, 2.7, 1.45]) # all default times > expiry, unsorted
# k = 2 # 2nd to default 
# pv_premium_leg = calc_premium_leg(expiry, default_times, payment_frequency, k, interest_rate_curve)
# print(f'PV Premium leg = pv_premium_leg = {pv_premium_leg}')
# 
# 
# # default_times = np.array([9, 3.5, 6, 5.5, 8]) 
# 
# 
# # 
# # # assign correlation matrix sigma
# # correlation_matrix = sigma_regular_dependence
# # 
# # correlated_uniform_sample = sampling_student_t_copula(correlation_matrix, 7, dimension=5, power_of_two = 4)
# # 
# # print(f'Correlated uniform sample shape = {correlated_uniform_sample.shape}')
# # =============================================================================


# =============================================================================
# TEST DEFAULT LEG COMPUTATION
# =============================================================================
print('***** TEST 9 ***** TEST DEFAULT LEG COMPUTATION ***** default_leg *****')
from functions import parse_interest_rate_curve, calc_default_leg

# general input arguments

expiry = 5 # in years
recovery_rate = 0.4

# get interest rate curve

work_dir = os.getcwd()
cur_dir = os.chdir('../..')  # move two directories up
data_set = 'final_basket'
data_set_directory = os.chdir("%s%s%s"%('data','/', data_set)) 
interest_rate_curve = parse_interest_rate_curve(data_set_directory, 
                                           'CDS_spreads_basket.xlsx', 
                                           'TEST_CURVE', 
                                           4, 
                                           'B:E', 
                                           ['Instr.Name', 'Close','START DATE','Mat.Dat'])

# =============================================================================
# print(' Case 1: All defaults occur after expiry')
# default_times = np.array([9, 7, 6, 5.5, 8]) # all default times after expiry, unsorted
# k = 2 # 2nd to default
# weights = [0.2, 0.2, 0.2, 0.2, 0.2] 
# pv_default_leg = calc_default_leg(expiry, default_times, recovery_rate, weights, k, interest_rate_curve)
# print(f'PV Default leg = pv_default_leg = {pv_default_leg}')
# =============================================================================

print(' Case 2: At least one default occurs before expiry')
default_times = np.array([9, 7, 3.2, 2.7, 1.49]) # at least one default occurs before expiry
k = 2 # 2nd to default
weights = [0.2, 0.2, 0.2, 0.2, 0.2] 
pv_default_leg = calc_default_leg(expiry, default_times, recovery_rate, weights, k, interest_rate_curve)
print(f'PV Default leg = pv_default_leg = {pv_default_leg}')



