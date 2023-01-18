#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 09:32:46 2022

@author: Stefan Mangold
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from math import log, exp

# functions

def get_days_between(past_date, future_date):
    '''
    

    Parameters
    ----------
    past_date : TYPE
        DESCRIPTION.
    future_date : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    difference = future_date - past_date
    return difference/ timedelta(days=1)

def sobol_sequence():
    '''
    

    Returns
    -------
    None.

    '''
    pass

def multivariate_random_normal(mu, sigma, num_simulations):
    '''
    

    Parameters
    ----------
    mu : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    num_simulations : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    pass

#sampler = qmc.Sobol(d=5, scramble=False)
#sample = sampler.random_base2(m=7)
#print(sample)

#print(qmc.discrepancy(sample))


#rng = np.random.default_rng(12345)
#print(rng)

#rfloat = rng.random(size=128)
#print(rfloat)


def parse_interest_rate_curve(data_set_dir, excel_file_name_, sheet_name_, header_offset, column_range, header_names):
    '''
    parses quotes of market instruments from a data provider, computes first ACT365 zero rates 
    and then ACT360 discount factors based on the ACT365 zero rates
    
    methodology described here:
        https://quant.stackexchange.com/questions/73522/how-does-bloomberg-calculate-the-discount-rate-from-eur-estr-curve
    
    Parameters
    ----------
    data_set_dir : string literal 
        directory where the Excel sheet to be parsed is located
    excel_file_name_ : string literal 
        DESCRIPTION. name of the excel file including type: 'CDS_spreads.xlsx' 
    sheet_name_ : string literal
        name of the sheet: 'ESTR'
    header_offset : integer number 
        offset from first row where the header row is located: 2
    column_range : string literal
        Excel range containing the data to be parsed: 'B:G'
    header_names : list of string literals
        list containing the column names to be parsed: ['Instr.Name', 'Close','START DATE',

    Returns
    -------
    ir_curve_frame : pandas dataframe containing interpolated discount factors

    '''
    ir_curve_frame = pd.read_excel(open(excel_file_name_, 'rb'),
                  sheet_name = sheet_name_, 
                  header = header_offset, 
                  usecols = column_range,
                  names = header_names)
    
    ir_curve_frame['Time To Maturity'] = get_days_between(ir_curve_frame['START DATE'], ir_curve_frame['Mat.Dat'])
    
    ir_curve_frame['Zero Rate ACT365'] = 365/(ir_curve_frame['Time To Maturity'])*np.log(1+(ir_curve_frame['Close']/100)*(ir_curve_frame['Time To Maturity']/360))

    ir_curve_frame['Discount Factor ACT360'] = np.exp(- ir_curve_frame['Zero Rate ACT365'] * ir_curve_frame['Time To Maturity']/365)
    
    return ir_curve_frame


def loglinear_discount_factor(maturity, discount_factor, tenor):
    '''
    - does log-linear interpolation of discount factors 
    based on pillar point input argument discount_factor 
    and at tenors specified by input argument tenor
    - does flat extrapolation beyond the min and max CDS maturity times
    - assumes that input maturity is sorted in ascending order

    Parameters
    ----------
    maturity : list of float
        [1, 15/12, 0.75, 21/12, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    discount_factor : list of float
        discount factors at tenor points of market quoted instruments
    tenor : list of float
        tenor points (time in years from t = now) at which we want to have discount factors, ACT/365 convention

    Returns
    -------
    df : list of floats
        log-linearly interpolated discount factors, ACT/365 convention

    '''
    
    max_time_index = len(maturity) - 1
    
    # corner cases
    if tenor == 0: df = 1.
    if tenor > 0 and tenor < maturity[0]: df = discount_factor[0]
    if tenor >= maturity[max_time_index]: df = discount_factor[max_time_index]
        
    # regular cases
    for i in range(0, max_time_index):
         if tenor >= maturity[i] and tenor < maturity[i+1]:
            term1 = ((tenor-maturity[i])/(maturity[i+1] - maturity[i]))*log(discount_factor[i+1])
            term2 = ((maturity[i+1]-tenor)/(maturity[i+1] - maturity[i]))*log(discount_factor[i])
            ln_df = term1 + term2
            df = exp(ln_df)
            
    return df
    

def cds_bootstrapper(maturity, discount_factor, spread, recovery, plot_prob=False, plot_hazard=False):
    '''
    Bootstrapping algorithm to extract implied hazard rates or implied survival probabilities.
    Uses a simplified CDS pricing approach, also known as the JP Morgan method.
    Has been shown to exactly match the results of CDS bootstrapping using the open source QuantLib library.

    Parameters
    ----------
    maturity : pandas dataframe column of float
        The maturities of the CDS contracts in years.
    discount_factor : pandas dataframe column of float
        Discount factors @ the maturities of the CDS contracts 
    spread : pandas dataframe column of float
        Quoted market spreads for the CDS contracts
    recovery : float
        the assumed recovery rate, a float value between 0.0 and 1.0
    plot_prob : boolean, optional
        Specifies whether a plot of survial shall be drawn. The default is False.
    plot_hazard : boolean, optional
        Specifies whether a plot of hazard rates shall be drawn. The default is False.

    Returns
    -------
    df : pandas dataframe
        fills the dataframe with survival / default probabilities, 
        hazard rates, marginal probability of default 
    '''
    
    # subsume list of inputs into a dataframe
    df = pd.DataFrame({'Maturity': maturity, 'Df': discount_factor, 'Spread': spread})
    
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


def t_copula_density(uniform_pseudo_sample_uni, nu, sigma):
    '''
    computes the t-copula density for one 
    uniform pseudo sample, a 1 x n column vector

    Parameters
    ----------
    uniform_pseudo_sample : TYPE
        DESCRIPTION.
    nu : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    


