#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 09:32:46 2022

@author: Stefan Mangold
"""

import os
import pandas as pd
import numpy as np
# Gamma function
from scipy.special import gamma
# Student-t distribution functions
from scipy.stats import t
from scipy import stats 
from scipy.stats import qmc # Sobol with direction numbers S. Joe and F. Y. Kuo
from functools import reduce
from datetime import timedelta, datetime
from math import log, exp

import matplotlib.pylab as plt

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
    #sampler = qmc.Sobol(d=5, scramble=False)
    #sample = sampler.random_base2(m=7)
    #print(sample)

    #print(qmc.discrepancy(sample))
    
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


def parse_pseudo_samples(data_set_directory_name, data_set_file_name, data_set_headers):
    '''
    Reads pseudo-sample data from a csv file into a pandas dataframe

    Parameters
    ----------
    data_set_directory_name : string literal
        name of directory containing pseudo sample csv file, e.g. 'PseudoSamples'
    data_set_file_name : string literal
        name of csv file containing pseudo sample data, including file type ending, eg. 'pseudo_samples.csv'
    data_set_headers : list of string literals
        a list of cds entity header names, e.g.  ['Deutsche Bank', 'Tesco', ...]

    Returns
    -------
    pseudo_sample_data_frame : pandas dataframe
        dataframe containing pseudo samples

    '''
    os.chdir('../..')  # move two directories up
    os.chdir("%s%s%s"%('data','/', data_set_directory_name)) 

    pseudo_sample_data_frame = pd.read_csv(data_set_file_name, 
                                           header=0, 
                                           names=data_set_headers)  
    
    return pseudo_sample_data_frame

def loglinear_discount_factor(maturity, discount_factor, tenor):
    '''
    - does log-linear interpolation of discount factors 
    based on pillar point input argument discount_factor 
    and at tenors specified by input argument tenor
    - does flat extrapolation beyond the min and max CDS maturity times
    - assumes that input maturity is sorted in ascending order

    Parameters
    ----------
    maturity (in years): list of float
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
    lower_idx = np.searchsorted(maturity, tenor)
    upper_idx = lower_idx + 1
    df = []
    
    for t_idx, t_val in enumerate(tenor):
        # corner cases
        if t_val == 0: 
            df.append(1.)
            continue
        
        if t_val in maturity:
            df.append(discount_factor[lower_idx[t_idx]])
            continue
        
        if t_val > 0 and t_val < maturity[0]: 
            df.append(discount_factor[0])
            continue
        
        if t_val >= maturity[max_time_index]: 
            df.append(discount_factor[max_time_index])
            continue
            
        # regular cases
        denominator =  maturity[upper_idx[t_idx]] - maturity[lower_idx[t_idx]]
        term1  = ((t_val-maturity[lower_idx[t_idx]]) / denominator) * log(discount_factor[upper_idx[t_idx]])
        term2 = ((maturity[upper_idx[t_idx]]-t_val) / denominator) * log(discount_factor[lower_idx[t_idx]])
        ln_df = term1 + term2
        df.append(exp(ln_df))
            
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

def linearise_spearman_correlation_matrix(spearman_corr_matrix):
    '''
    applies 'linearisation' to a spearman correlation matrix
    e.g. based on linear historical correlation of near uniformly
    distributed pseudo-samples
    

    Parameters
    ----------
    spearman_corr_matrix : 
        a square spearman correlation matrix

    Returns
    -------
    linearised_corr_matrix : np.array of floats
        a linearised spearman correlation matrix m(i,j), 
        where all off diagonal elements are linearised
        by applying the function 2 x sin(pi / 6 * m(i, j))
        to all (i, j) where i not equal j

    '''
    
    # create mask for off-diagonal elements
    mask = ~np.eye(spearman_corr_matrix.shape[0],dtype=bool)
    # apply linearization to all off-diagonal elements
    linearised_corr_matrix = np.where(mask,(2* np.sin(spearman_corr_matrix * np.pi / 6) ).astype(float), spearman_corr_matrix)

    return linearised_corr_matrix

    



def student_t_copula_density(uniform_pseudo_sample, n, nu, f1, f2, f3, sigma_det, sigma_inverse):
    '''
    computes the Student-t copula density for one 
    uniform pseudo sample, a 1 x n column vector
    of uniformly distributed samples 

    Parameters
    ----------
    uniform_pseudo_sample : np.array (1 x n)
        a 1 x n column array of uniform samples
    n : int
        size of pseudo-sample (e.g. 5)
    nu : int
        degree of freedom parameter for Student-t distribution
    f1 : float
        first term of the Student-t probability density formula
        1. / np.sqrt(sigma_det)
    f2: float
        second term of the Student-t probability density formula
        gamma( (\nu + n) / 2) / gamma( \nu / 2)
    f3: float
        third term of the Student-t probability density formula
        (gamma( \nu / 2 ) / gamma( (\nu + 1) / 2)) ^ n
    sigma_det : float
        determinant of rank correlation matrix sigma
    sigma_inverse : (n x n) np.array (2d)
        inverse of rank correlation matrix sigma

    Returns
    -------
    probability density for one pseudo-sample

    '''    
    inv_cdf = np.matrix(t.ppf(uniform_pseudo_sample, nu)) # wrap in np.matrix to enable transpose
    
    f4_num_fraction = (inv_cdf * sigma_inverse * inv_cdf.T) / nu
    f4_num = np.power(1. + f4_num_fraction, -(nu + n)/2.)
    
    f4_denom_list = [(1. + (t.ppf(u, nu)**2) / nu) for u in uniform_pseudo_sample]
    f4_denom = (reduce((lambda x, y: x * y), np.power(f4_denom_list, -(nu + 1.) / 2.)))
    
    density = f1 * f2 * f3 * f4_num / f4_denom
    
    return density.item()
    
    
def student_t_loglikelihood(pseudo_samples, nu, sigma):
    '''
    

    Parameters
    ----------
    pseudo_samples : pandas dataframe
        pseudo samples, uniformly distributed, each row is one sample
    nu : int
        degree of freedom parameter for Student-t distribution
    sigma : np.array of dimension n x n
        n x n (symmetric) correlation matrix 

    Returns
    -------
    loglikelihood : float
        estimation of the loglikelihood function to estimate
        the degree-of-freedom parameter nu

    '''

    n = pseudo_samples.shape[1]
    # compute determinant and inverse of correlation matrix sigma once
    
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    
    # compute all components of the Student-t density formula that do not depend on the pseudo sample
    f1 = 1. / np.sqrt(sigma_det)
    f2 = gamma((nu + n )/ 2.) / gamma(nu / 2.)
    f3 = np.power(gamma(nu / 2.) / gamma((nu + 1.) / 2.), n)
    
    np_pseudo_samples = pseudo_samples.to_numpy()
    list_of_log_densities = [np.log(student_t_copula_density(this_sample, n, nu, f1, f2, f3, sigma_det, sigma_inv)) for this_sample in np_pseudo_samples]
    loglikelihood = reduce((lambda x, y: x + y), list_of_log_densities)

    return loglikelihood

def maximum_likelihood_student_t_dof(pseudo_samples, sigma, plot_likelihood=False):
    '''
    Implements maximum likelihood estimation of Student-t distribution
    degrees of freedom parameter \nu by varying \nu from 
    1 to 30 and computing the corresponding log likelihood value 

    Parameters
    ----------
    pseudo_samples : pandas dataframe
        pseudo samples, uniformly distributed, each row is one sample
    sigma : np.array of dimension n x n
        n x n (symmetric) correlation matrix
    plot_likelihood : BOOLEAN, optional
        boolean to indicate whether a plot shall be drawn. 
        The default is False.

    Returns
    -------
    maximum_likelihood : TYPE
        DESCRIPTION.

    '''

    parameter_space_nu = list(range(1, 31))
    
    maximum_likelihood = { nu : student_t_loglikelihood(pseudo_samples, nu, sigma) for nu in parameter_space_nu}

    if plot_likelihood:
        lists = sorted(maximum_likelihood.items()) 
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.show()

    return maximum_likelihood    


def sampling_gaussian_copula(sigma, dimension=5, power_of_two=7):
    '''
    Implementation of sampling from Gaussian copula.
    Returns the entire sample of correlated uniformly
    distributed random variables.
    The sample size must be a power of two in order for 
    the Sobol sequence to keep its balance properties.
    Sobol numbers are scrambled to avoid drawing exactly zero
    and to avoid numerical division by zero problems. 
    
    Parameters
    ----------
    sigma : Array of float
        Rank or linear correlation matrix. Needs to be positive & semi-definite,
        if not Cholesky decomposition will fail and raise a
        LinAlgError     
    dimension : int, optional. Default is 5.
        Number of stochastic variables to be drawn simultaneously.
    power_of_two : int, optional. Default is 7.
        determines the sample size, which is 2 ^ power_of_two. 
        Per default returning a sample size of 2^7 = 128 samples

    Returns
    -------
    correlated_uniform_rvs : float
        correlated uniform random variables of dimension n x sample size
        n : number of entities in the default basket
        sample size is: 2 ** power_of_two

    '''
    
    # 1. Do Cholesky factorization, obtain decomposed matrix A
    cholesky_matrix_A = np.linalg.cholesky(sigma)
    
    # 2. Sample independent uniformly distributed variables U
    sobol_object_5d = qmc.Sobol(d=dimension, scramble=True)
    sobol_uniform_rvs = sobol_object_5d.random_base2(m=power_of_two)
    
    # 3. Convert uniforms U from step 2. into Normal random variables Z
    standard_normal_rvs = stats.norm().ppf(sobol_uniform_rvs)
    
    # 4. Convert into correlated normal X using X = AZ
    correlated_normal_rvs = np.matmul(cholesky_matrix_A, standard_normal_rvs.T)
    
    # 5. Convert to correlated uniform vector by U = Phi(X), Phi being the standard normal CDF 
    correlated_uniform_rvs = stats.norm().cdf(correlated_normal_rvs.T)
    
    return correlated_uniform_rvs


def sampling_student_t_copula(sigma, nu, dimension=5, power_of_two=7):
    '''
    Implementation of sampling from Student-t copula.
    Returns the entire sample of correlated uniformly
    distributed random variables.
    The sample size must be a power of two in order for 
    the Sobol sequence to keep its balance properties.
    Sobol numbers are scrambled to avoid drawing exactly zero
    and to avoid numerical division by zero problems. 
    
    Parameters
    ----------
    sigma : Array of float
        Correlation matrix needs to be a rank correlation matrix. 
        Needs to be positive and semi-definite;
        if not Cholesky decomposition will fail and raise a numpy
        LinAlgError
    nu : int
        degrees of freedom parameter of Student-t distribution
    dimension : int, optional. Default is 5.
        Number of stochastic variables to be drawn simultaneously.
    power_of_two : int, optional. Default is 7.
            determines the sample size, which is 2 ^ power_of_two. 
            Per default returning a sample size of 2^7 = 128 samples
    


    Returns
    -------
    correlated_uniform_rvs : float
        correlated uniform random variables of dimension n x sample size
        n : number of entities in the default basket
        Per default returning a sample size of 2^7=128 samples

    '''

    # 1.1 apply linearization to off-diagonal elements before decomposition
   
    # 1.2 Compute decomposition of correlation matrix sigma = A * A^T
    cholesky_matrix_A = np.linalg.cholesky(sigma) 
    
    # 2.1 Sample independent uniformly distributed variables U
    sobol_object_5d = qmc.Sobol(d=dimension, scramble=True)
    sobol_uniform_rvs = sobol_object_5d.random_base2(m=power_of_two)
    
    # 2.2 Draw an n-dimensional vector of independent standard Normal variables Z = (z_1, ... , z_n)^T
    standard_normal_rvs = stats.norm().ppf(sobol_uniform_rvs)
    
    # 3. Draw independent chi-squared random variables s ~ Chi^2_nu
    # 3.1 Draw a nu-dimensional vector of uniforms
    sobol_object_nu_d = qmc.Sobol(d=nu, scramble=True)
    sobol_uniform_2_normal_rvs = sobol_object_nu_d.random_base2(m=power_of_two)
    
    # 3.2 Convert to standard Normal
    standard_normal_2_chi2_rvs = stats.norm().ppf(sobol_uniform_2_normal_rvs)
    del(sobol_uniform_2_normal_rvs)
    
    # 3.3 Square and sum
    chi2_rvs = np.sum(np.power(standard_normal_2_chi2_rvs, 2), axis=1)
    del(standard_normal_2_chi2_rvs)
    
    # 4. Compute n-dimensional Student-t vector Y = Z / sqrt(s / nu)
    student_t_rvs = standard_normal_rvs / np.sqrt(chi2_rvs[:, None] / nu)
    
    # 5. Impose correlation by X = AY
    correlated_normal_rvs = np.matmul(cholesky_matrix_A, student_t_rvs.T)
    
    # 6. Map to a correlated uniform vector by U = T_nu (X) using the CDF of Student-t distribution
    correlated_uniform_rvs = stats.t.cdf(correlated_normal_rvs.T, nu)

    return correlated_uniform_rvs
    
    
    
def calc_premium_leg(expiry, default_times, payment_frequency, k, interest_rate_curve):
    '''
    Computes the present value of the premium leg of the basket CDS.

    Parameters
    ----------
    expiry : float
        as seen from the valuation date, the time to expiry denominated in years / year fractions
    default_times : float
        as seen from the valuation date, the default times denominated in years / year fractions
    payment_frequency : int
        the number of times per annum premium payments are made (4 == quarterly payments)
    k  : int
        number of protected defaults.        
    interest_rate_curve : pandas dataframe
        dataframe containing zero rates, discount factors, etc.
       
    Returns
    -------
    pv_premium_leg : float
        present value of the premium leg

    '''
    
    pv_premium_leg = 0
    current_notional = 1
    size_basket = 5
    
    num_defaults = np.count_nonzero(default_times < expiry)
    
    # sort default times in increasing order
    if num_defaults > 0:
        default_times = np.sort(default_times)
    
    #### TO BE MOVED OUTSIDE OF THIS FUNCTION AND PASSED AS ARGUMENT
    # number of time points
    grid_size = expiry * payment_frequency
    
    # time grid
    time_grid = np.array([ i / payment_frequency for i in range (0, grid_size + 1)])
    
    # the time steps 'dt'
    time_delta = np.diff(time_grid, n=1, axis=0)
    ####
    
    # discount factors at time grid points
    
    #### TO BE MOVED OUTSIDE OF THIS FUNCTION AND PASSED AS ARGUMENT
    discount_factors = np.array([loglinear_discount_factor(interest_rate_curve['Time To Maturity']/365, interest_rate_curve['Discount Factor ACT360'], t) for t in time_grid])
    ####
    
    # Case 1: all default times are after expiry
    if num_defaults == 0:
        arguments = time_delta, discount_factors[1:]
        pv_premium_leg = sum(reduce(lambda a, b: a * b, data) for data in zip(*arguments))
        return pv_premium_leg
        
    
    # Cases 2 and 3: one or more defaults occur before expiry
    default_times_indices = np.sort(time_grid.searchsorted(default_times))   
    
    # notional grid           
    notional_structure = np.ones(grid_size)

    # reduce notional proportionally for each occured default before expiry
    # assumes that each name has the same notional (or weight) in the basket
    for this_default_time_index in default_times_indices[0:k-1]:
        if this_default_time_index < grid_size:
            current_notional -= 1/size_basket
            notional_structure[this_default_time_index-1:] = current_notional
        else:
            break
        
    # Case 2: all defaults are protected, that is: num_defaults <= k, no early expiry   
    if num_defaults <= k: 
        arguments = time_delta, discount_factors[1:], notional_structure
        pv_premium_leg = sum(reduce(lambda a, b: a * b, data) for data in zip(*arguments))
    # Case 3: number of defaults before expiry is higher than number of protected defaults: num_defaults > k, early expiry 
    else:   
        last_protected_default_time_index = default_times_indices[k-1]
        arguments = time_delta[0:last_protected_default_time_index], discount_factors[1:last_protected_default_time_index+1], notional_structure[0:last_protected_default_time_index]
        pv_premium_leg = sum(reduce(lambda a, b: a * b, data) for data in zip(*arguments))
        
    return pv_premium_leg

def calc_premium_leg_fast(default_times_sorted, num_defaults, payment_frequency, k, interest_rate_curve, grid_size, time_grid, time_delta, discount_factors):
    '''
    Fast version of the premium leg computation of the basket CDS.
    Precomputes inputs such as time grid and discount factors to speed up computations.

    Parameters
    ----------
    default_times_sorted : TYPE
        DESCRIPTION.
    num_defaults : TYPE
        DESCRIPTION.
    payment_frequency : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    interest_rate_curve : TYPE
        DESCRIPTION.
    grid_size : TYPE
        DESCRIPTION.
    time_grid : TYPE
        DESCRIPTION.
    time_delta : TYPE
        DESCRIPTION.
    discount_factors : TYPE
        DESCRIPTION.

    Returns
    -------
    pv_premium_leg : float
        present value of the premium leg

    '''
    
    pv_premium_leg = 0
    current_notional = 1
    size_basket = 5
            
    # Case 1: all default times are after expiry
    if num_defaults == 0:
        arguments = time_delta, discount_factors[1:]
        pv_premium_leg = sum(reduce(lambda a, b: a * b, data) for data in zip(*arguments))
        return pv_premium_leg
        
    
    # Cases 2 and 3: one or more defaults occur before expiry
    default_time_indices = np.sort(time_grid.searchsorted(default_times_sorted))   
    
    # notional grid           
    notional_structure = np.ones(grid_size)

    # reduce notional proportionally for each occured default before expiry
    # assumes that each name has the same notional (or weight) in the basket
    for this_default_time_index in default_time_indices[0:k-1]:
        if this_default_time_index < grid_size:
            current_notional -= 1/size_basket
            notional_structure[this_default_time_index-1:] = current_notional
        else:
            break
        
    # Case 2: all defaults are protected, that is: num_defaults <= k, no early expiry   
    if num_defaults <= k: 
        arguments = time_delta, discount_factors[1:], notional_structure
        pv_premium_leg = sum(reduce(lambda a, b: a * b, data) for data in zip(*arguments))
    # Case 3: number of defaults before expiry is higher than number of protected defaults: num_defaults > k, early expiry 
    else:   
        last_protected_default_time_index = default_time_indices[k-1]
        arguments = time_delta[0:last_protected_default_time_index], discount_factors[1:last_protected_default_time_index+1], notional_structure[0:last_protected_default_time_index]
        pv_premium_leg = sum(reduce(lambda a, b: a * b, data) for data in zip(*arguments))
        
    return pv_premium_leg
    
    
    
    

def calc_default_leg(expiry, default_times, recovery_rate, weights, k, interest_rate_curve):
    '''
    Computes the present value of the default leg of the basket CDS.
    
    Parameters
    ----------
    expiry : float
        as seen from the valuation date, the time to expiry denominated in years / year fractions
    default_times : float
        as seen from the valuation date, the default times denominated in years / year fractions
    recovery_rate : float
        the fraction of the loss recovered in case of default as a percentage 
    weights : TYPE
        DESCRIPTION.
    k  : int
        number of protected defaults. 
    interest_rate_curve : pandas dataframe
        dataframe containing zero rates, discount factors, etc.

     Returns
     -------
     pv_default_leg : float
         present value of the default leg

    '''
    num_defaults = np.count_nonzero(default_times < expiry)
    # sort default times in increasing order
    if num_defaults > 0:
        default_times = np.sort(default_times)
        
    kth_default_idx = k - 1
    kth_default_time = default_times[kth_default_idx]
    
    if (num_defaults == 0):
        pv_default_leg = 0
    else:
        discount_factor = loglinear_discount_factor(interest_rate_curve['Time To Maturity']/365, interest_rate_curve['Discount Factor ACT360'], kth_default_time)
        pv_default_leg = (1. - recovery_rate) * weights[kth_default_idx] * discount_factor
    
    
    return pv_default_leg
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
