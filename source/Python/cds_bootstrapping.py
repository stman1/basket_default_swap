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

# Plot settings
matplotlib.rcParams['figure.figsize'] = [14.0, 8.0]
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['lines.linewidth'] = 2.0

# Interactive plotting
import cufflinks as cf
cf.set_config_file(offline=True)
#cf.set_config_file(theme = 'pearl')
# cf.getThemes() ['ggplot', 'pearl', 'solar', 'space', 'white', 'polar', 'henanigans']


# functions
def get_discount_factor(maturity, discountfactor, tenor):
    
    max_time_index = len(maturity) - 1
    
    if tenor == 0: Df = 1.
    if tenor > 0 and tenor < maturity[0]: Df = discountfactor[0]
    if tenor >= maturity[max_time_index]: Df = discountfactor[max_time_index]
        
    for i in range(0, max_time_index):
         if tenor >= maturity[i] and tenor < maturity[i+1]:
            term1 = ((tenor-maturity[i])/(maturity[i+1] - maturity[i]))*log(discountfactor[i+1])
            term2 = ((maturity[i+1]-tenor)/(maturity[i+1] - maturity[i]))*log(discountfactor[i])
            lnDf = term1 + term2
            Df = exp(lnDf)
            
    return Df

def get_survival_probability(maturity, discountfactor, spread, recovery, plot_prob=False, plot_hazard=False):
    
    # subsume list of inputs into a dataframe
    df = pd.DataFrame({'Maturity': maturity, 'Df': discountfactor, 'Spread': spread})
    
    # convert bps to decimal
    df['Spread'] = df['Spread']/10000

    # specify delta_t
    df['Dt'] = df['Maturity'].diff().fillna(0)

    # loss rate
    L = 1.0 - recovery
    
    # initialize the variables
    term = term1 = term2 = divider = 0
    
    for i in range(0, len(df.index)):
        if i == 0: df.loc[i,'Survival'] = 1; df.loc[i, 'Hazard'] = 0
        if i == 1: df.loc[i,'Survival'] = L / (L+df.loc[i,'Dt']*df.loc[i,'Spread']); \
            df.loc[i, 'Hazard'] = -log(df.loc[i,'Survival']/df.loc[i-1,'Survival'])/df.loc[i,'Dt']
        if i > 1:
            terms = 0
            for j in range(1, i):
                term = df.loc[j,'Df']*(L*df.loc[j-1,'Survival'] - \
                                              (L + df.loc[j,'Dt']*df.loc[i,'Spread'])* \
                                              df.loc[j,'Survival'])
                terms = terms + term  
           
            divider = df.loc[i,'Df']*(L+df.loc[i,'Dt']*df.loc[i,'Spread'])
            term1 = terms/divider

            term2 = (L*df.loc[i-1,'Survival']) / (L + (df.loc[i,'Dt'] * df.loc[i,'Spread']))

            df.loc[i,'Survival'] = term1 + term2
            
            if (df.loc[i,'Survival'] >= 0 and df.loc[i-1,'Survival'] >= 0):
                df.loc[i, 'Hazard'] = -log(df.loc[i,'Survival']/df.loc[i-1,'Survival'])/df.loc[i,'Dt']
    
    # derive probability of default
    df['Default'] = 1. - df['Survival']
    
    # derive marginal probability of default
    df['Marginal'] = df['Survival'].diff().fillna(0)
    
    if plot_prob:
        # plot survival probability
        df[['Survival', 'Default']].iplot(title='Survival vs Default Probability', 
                                          xTitle='CDS Maturity', 
                                          yTitle='Survival Probability', 
                                          secondary_y = 'Default', 
                                          secondary_y_title='Default Probability')
        
    if plot_hazard:
        # plot survival probability
        df['Hazard'].iplot(kind='bar', title='Term Structure of Hazard Rates', 
                                          xTitle='CDS Maturity', 
                                          yTitle='Hazard Rates')

    return df


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
    Df[i] = get_discount_factor(cds_df.Maturity,cds_df.DiscountFactors,t[i])
    

print(f'Discount Factors: {Df}')


# interpolation Prudential spreads
prudential = np.interp(t,cds_df.Maturity,cds_df.Prudential)
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

prudential_survival_p = get_survival_probability(df.Maturity,df.Df,df.Prudential,0.40,plot_prob=True,plot_hazard=True)
prudential_survival_p



