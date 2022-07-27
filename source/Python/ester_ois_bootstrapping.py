#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:30:23 2022

@author: stefanmangold
"""
import QuantLib as ql


# adapted to €STR-discounting from https://quant.stackexchange.com/questions/58333/bootstrapping-ois-curve

tenors = [
    '1D', '1W', '2W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M', '10M', '11M', '1Y',
     '18M', '2Y', '30M', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y',  '11Y', '12Y',
     '15Y', '20Y', '25Y', '30Y', '35Y', '40Y', '50Y']

rates = [
    -0.467, -0.472, -0.47, -0.46, -0.471, -0.47, -0.481, -0.487, -0.5, -0.495, -0.5, -0.506,
     -0.51, -0.515, -0.52, -0.541, -0.551, -0.556, -0.56, -0.551, -0.531, -0.5, -0.462, -0.426,
    -0.379, -0.337, -0.293, -0.251, -0.147, -0.068, -0.055, -0.09, -0.099, -0.134, -0.172]

eonia = ql.Eonia()
helpers = []
for tenor, rate in zip(tenors,rates):
    if tenor == '1D':
        helpers.append( ql.DepositRateHelper(rate / 100, eonia ) )
    else:
        helpers.append( ql.OISRateHelper(2, ql.Period(tenor), ql.QuoteHandle(ql.SimpleQuote(rate/100)), eonia) )
        
eonia_curve = ql.PiecewiseLogCubicDiscount(0, ql.TARGET(), helpers, ql.Actual365Fixed()) 
discount_curve = ql.YieldTermStructureHandle(eonia_curve)
swapEngine = ql.DiscountingSwapEngine(discount_curve)


overnightIndex = ql.Eonia(discount_curve)
for tenor, rate in zip(tenors, rates):
    if tenor == '1D': continue
    ois_swap = ql.MakeOIS(ql.Period(tenor), overnightIndex, 0.01, pricingEngine=swapEngine)
    print(f"{tenor}\t{ois_swap.fairRate():.4%}\t{rate:.4f}%")
