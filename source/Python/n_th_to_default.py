#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from enum import Enum
import numpy as np
import scipy as sc
from numpy import matlib
from scipy.stats import norm
"""
Created on Tue Jul 5 18:15:55 2022

@author: stefanmangold
"""

class ModelType(Enum):
    MONTE_CARLO = 1
    FFT = 2
    HULL_WHITE = 3


def default_until_t(time_t, default_times):
    """
    The function default_until_t records default until time t
    time_t :        time as numerical offset year_fraction from start date
    default_times:  default times
    
    returns a matrix of zeros and ones indicating defaults per simulation and per 
    """
    num_rows, num_cols = default_times.shape
    nk = np.maximum(default_times - np.ones([num_rows, num_cols]) * time_t < 0, 0)    
    return nk


def n_th_to_default(model, t_mat, rho, hazard, int_rate, rec, k, pay_freq_pa, num_sim):
    """
    The function n_th_to_default calculates the fair spread per annum (in basis points)
    paid f times per year for a k-th to default basket swap
    
    model can be 'MONTE_CARLO', 'FFT' or 'HULL_WHITE';
    
    t_mat: maturity of the contract
    rho: 1xn vector of the correlations to the factor
    hazard: 1xn vector of the hazard rates
    rate: interest rate
    rec: common recovery rate
    k: contract specifier (1st, 2nd, 3rd..nth to default)
    pay_freq_pa: interannual number of payments
    num_sim: the number of Monte Carlo simulations
    """
    
    if rho < 1:
        if model == ModelType.MONTE_CARLO:
            pass
            # Premium Leg
            
            tau = gaussian_copula_default_times(rho, hazard, num_sim)
            pv_premium_leg = np.zeros([1, num_sim])
            num_observations = t_mat * pay_freq_pa
            for obs in range(1, num_observations, 1):
                A = np.maximum((np.sum(default_until_t(obs / pay_freq_pa, tau), axis = 0) - np.ones([1, num_sim]) * k) < 0, 0)
                pv_premium_leg += A * np.exp(-int_rate * (obs / pay_freq_pa))
    

            # Default Leg
            
            tau = np.vstack((np.ones([1, num_sim]) * np.Inf, np.sort(tau, axis = 0)))
            A = (np.max((np.sum(default_until_t(t_mat, tau), axis = 0) - np.ones([1, num_sim]) * k ) >= 0, 0) * k) 
            temp = np.take_along_axis(-tau, A[None, :], axis=0)
            pv_default_leg = np.exp(np.take_along_axis(-tau, A[None, :], axis=0) * int_rate) * (1 - rec);
            fair_spread = 10000 * (np.sum(pv_default_leg) / np.sum(pv_premium_leg * pay_freq_pa));
        else:
            pass
        
    
    else:
        dsc = np.sort(hazard)
        fair_spread = 10000 * (1 - rec) * dsc[k]
    return fair_spread


def gaussian_copula_default_times(rho, h, sim):
    n = len(h)
    h = np.transpose(np.array([h]))
    sigma = np.ones([n, n]) * rho**2 + np.diag(np.ones(n)) * (1-rho**2)
    x = np.transpose(np.random.multivariate_normal(np.zeros(n), sigma, sim))
    u = norm.cdf(x)
    tau = -matlib.repmat(np.power(h, -1), 1, sim) * np.log(u)
    return tau


hazard_rates = [0.01, 0.01, 0.01, 0.01, 0.01]

rho = 1

#default_times = gaussian_copula_default_times(rho, h, 100000)
#print(default_times)

time_to_maturity = 5
interest_rate = 0.05
recovery_rate = 0.5
payment_frequency_per_annum = 1
number_simulations = 100000
to_default = 1

fair_spread = n_th_to_default(ModelType.MONTE_CARLO, time_to_maturity, rho, hazard_rates, interest_rate, recovery_rate, to_default, payment_frequency_per_annum, number_simulations)

#nk = default_until_t(time, default_times)
#print(nk)


#function tau = GCdef(rho,h,sim)

#randn('state',10) % fixed seed
#N = length(h);
#S = ones(N,N).*(rho^2) + diag(ones(N,1),0).*(1 -(rho^2) );


#% Gaussian Copula Simulation
#x = MVNRND(zeros(1,N),S,sim)'; 
#u = normcdf(x);
#tau = -repmat((h.^-1)',1,sim).*log(u);

