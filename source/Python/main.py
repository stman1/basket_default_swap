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

# LOAD INTEREST RATE INFORMATION AND DISCOUNT FACTORS

# =============================================================================
# work_dir = os.getcwd()
# cur_dir = os.chdir('../..')  # move two directories up
# data_set = 'final_basket'
# data_set_directory = os.chdir("%s%s%s"%('data','/', data_set)) 
# ir_data_frame = parse_interest_rate_curve(data_set_directory, 
#                                           'CDS_spreads_basket.xlsx', 
#                                           'ESTR', 
#                                           4, 
#                                           'B:E', 
#                                           ['Instr.Name', 'Close','START DATE','Mat.Dat'])
# 
# 
# # TEST OF LOG-LINEAR INTERPOLATION OF DISCOUNT FACTORS
# 
# maturity = [0.5/12, 0.75/12, 1/12, 1/6, 1/4, 1/3, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 15/12, 0.75, 21/12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30
#             ]
# t = np.arange(29)
# cds_discount_factors = np.zeros(len(t))
# 
# for i in range(0, len(t)):
#     cds_discount_factors[i] = loglinear_discount_factor(maturity, ir_data_frame['Discount Factor ACT360'], t[i])
#     
# print(f'discount factors: {cds_discount_factors}')
# 
# =============================================================================

# LOAD PSEUDO SAMPLES FROM FILE
work_dir = os.getcwd()
cur_dir = os.chdir('../..')  # move two directories up
data_set = 'PseudoSamples'
data_set_directory = os.chdir("%s%s%s"%('data','/', data_set)) 

pseudo_sample_data_frame = pd.read_csv('pseudo.samples.csv', 
                                       header=0, 
                                       names= ['Prudential','BMW','Volkswagen','Deutsche Bank', 'Kering'])  

# COMPUTE CORRELATION MATRIX

# correlation matrix for Gaussian copula

# correlation matrix for Student-t copula


# linear correlation of uniform pseudo samples (Spearman correlation)
pseudo_sample_data_frame.corr() 

# SAMPLING FROM COPULA ALGORITHM


# TEST SAMPLING FROM STUDENT-T COPULA
print(f'***** TEST 9 ***** TEST BOOTSTRAPPING HAZARD RATES ***** cds_bootstrapper *****')
from functions import cds_bootstrapper, loglinear_discount_factor

# =============================================================================
# cds_bootstrapper(maturity, discount_factor, spread, recovery, plot_prob=False, plot_hazard=False):
#     '''
#     Bootstrapping algorithm to extract implied hazard rates or implied survival probabilities.
#     Uses a simplified CDS pricing approach, also known as the JP Morgan method.
#     Has been shown to exactly match the results of CDS bootstrapping using the open source QuantLib library.
# 
#     Parameters
#     ----------
#     maturity : pandas dataframe column of float
#         The maturities of the CDS contracts in years.
#     discount_factor : pandas dataframe column of float
#         Discount factors @ the maturities of the CDS contracts 
#     spread : pandas dataframe column of float
#         Quoted market spreads for the CDS contracts
#     recovery : float
#         the assumed recovery rate, a float value between 0.0 and 1.0
#     plot_prob : boolean, optional
#         Specifies whether a plot of survial shall be drawn. The default is False.
#     plot_hazard : boolean, optional
#         Specifies whether a plot of hazard rates shall be drawn. The default is False.
# 
#     Returns
#     -------
#     df : pandas dataframe
#         fills the dataframe with survival / default probabilities, 
#         hazard rates, marginal probability of default 
#     '''
# =============================================================================

maturity = 5
recovery = 0.4

t = [1, 2, 3, 4, 5]
prud = [29.5, 40.13, 50.6, 63.16, 74.18]
bmw = [28, 37.92, 47.09, 58.16, 70.2]
vw = [69.29, 81.66, 97, 111.93, 131.64]
db = [85.5, 91.32, 97, 103.49, 111.45]
ker = [13.28, 18.65, 24.05, 31.15, 38.24]
df = loglinear_discount_factor(maturity, discount_factor, tenor)
cds_df = pd.DataFrame({'Maturity' : t, 'prudential_spreads' : prud, 'bmw_spreads' : bmw, 'volkswagen_spreads' : vw, 'deutsche_bank' : db, 'kering' : ker, 'discount_factor': df})


spreads_prudential = cds_bootstrapper(cds_df.Maturity, cds_df.discount_factor, cds_df.prudential_spreads, recovery)


# assign correlation matrix sigma
#correlation_matrix = sigma_regular_dependence

#correlated_uniform_sample = sampling_student_t_copula(correlation_matrix, 7, dimension=5, power_of_two = 4)

#print(f'Correlated uniform sample shape = {correlated_uniform_sample.shape}')

