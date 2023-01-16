# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:23:53 2023

@author: Stefan Mangold
"""

# Import libraries
import pandas as pd
import numpy as np
from numpy import *

# Plotting library
import matplotlib
import matplotlib.pyplot as plt

# import user-defined functions
from functions import loglinear_discount_factor, cds_bootstrapper

# Plot settings
matplotlib.rcParams['figure.figsize'] = [14.0, 8.0]
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['lines.linewidth'] = 2.0

# Interactive plotting
import cufflinks as cf
cf.set_config_file(offline=True)
#cf.set_config_file(theme = 'pearl')
# cf.getThemes() ['ggplot', 'pearl', 'solar', 'space', 'white', 'polar', 'henanigans']


# global variables
recovery_rate = 0.40



# subsume list of inputs into a dataframe
cds_df = pd.DataFrame({'Maturity': [1, 2, 3, 4, 5], 
                   'Prudential': [29.5, 40.13, 50.6, 63.16, 74.18],
                   'Volkswagen': [69.29, 81.66, 97, 111.93, 131.64],
                   'BMW': [28, 37.92, 47.09, 58.16, 70.2],
                   'DeutscheBank': [85.5, 91.32, 97, 103.49, 111.45],
                   'Kering': [13.28, 18.65, 24.05, 31.15, 38.24],
                   'DiscountFactors': [0.96959, 0.93906, 0.91391, 0.88991, 0.86641]})
# output
cds_df


t = np.arange(6)
Df = np.zeros(len(t))
for i in range(0, len(t)):
    Df[i] = loglinear_discount_factor(cds_df.Maturity,cds_df.DiscountFactors,t[i])
    

print(f'Discount Factors: {Df}')


# spread interpolation Prudential spreads
prudential = np.interp(t, cds_df.Maturity, cds_df.Prudential)
# set spreads to zero at t=0
prudential[0] = 0

# interpolation remaining CDS spreads
vw = np.interp(t,cds_df.Maturity,cds_df.Volkswagen)
vw[0] = 0
bmw = np.interp(t,cds_df.Maturity,cds_df.BMW)
bmw[0] = 0
db = np.interp(t,cds_df.Maturity,cds_df.DeutscheBank)
db[0] = 0
kering = np.interp(t,cds_df.Maturity,cds_df.Kering)
kering[0] = 0


# output the results
print(f'Prudential Spreads: \t {prudential}')
print(f'Volkswagen Spreads: \t {vw}')
print(f'BMW Spreads: \t {bmw}')
print(f'Deutsche Bank Spreads: \t {db}')
print(f'Kering Spreads: \t {kering}')

# subsume list of inputs into a dataframe
df = pd.DataFrame({'Maturity': t, 
                   'Prudential': prudential,
                   'Volkswagen': vw,
                   'BMW': bmw,
                   'Deutsche Bank': db,
                   'Kering': kering,
                   'Df': Df})
# output
df

prudential_bootstrapped = cds_bootstrapper(df.Maturity,
                                         df.Df,
                                         df.Prudential,
                                         recovery_rate, 
                                         plot_prob=True, 
                                         plot_hazard=True)

print(f'Prudential Bootstrapped hazard rates: \n {prudential_bootstrapped}')

volkswagen_bootstrapped = cds_bootstrapper(df.Maturity,
                                         df.Df,
                                         df.Volkswagen,
                                         recovery_rate, 
                                         plot_prob=True, 
                                         plot_hazard=True)

print(f'Volkswagen Bootstrapped hazard rates: \n {volkswagen_bootstrapped}')

bmw_bootstrapped = cds_bootstrapper(df.Maturity,
                                         df.Df,
                                         df.BMW,
                                         recovery_rate, 
                                         plot_prob=True, 
                                         plot_hazard=True)

print(f'BMW Bootstrapped hazard rates: \n {bmw_bootstrapped}')

db_bootstrapped = cds_bootstrapper(df.Maturity,
                                         df.Df,
                                         df['Deutsche Bank'],
                                         recovery_rate, 
                                         plot_prob=True, 
                                         plot_hazard=True)

print(f'Deutsche Bank Bootstrapped hazard rates: \n {db_bootstrapped}')

kering_bootstrapped = cds_bootstrapper(df.Maturity,
                                       
                                         df.Df,
                                         df.Kering,
                                         recovery_rate, 
                                         plot_prob=True, 
                                         plot_hazard=True)

print(f'Kering Bootstrapped hazard rates: \n {kering_bootstrapped}')



