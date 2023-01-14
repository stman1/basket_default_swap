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
    ir_curve_frame : pandas dataframe
        parses quotes of market instruments from a data provider, computes ACT365 zero rates and ACT360 discount factors 

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
    
    

