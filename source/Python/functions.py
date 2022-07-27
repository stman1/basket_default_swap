#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:32:46 2022

@author: stefanmangold
"""

import os
import pandas as pd

work_dir = os.getcwd()

cur_dir = os.chdir('../..')

data_dir = os.chdir('data')
cds_spreads_df = pd.read_excel(open('default_basket_data.xlsx', 'rb'),
              sheet_name='CDS_spreads_history', 
              header = 20, 
              usecols = 'E:N',
              names=['Time Stamp','CREDIT SUISSE','BANCO SANTANDER', 'BNP', 'DEUTSCHE BANK', 'SOCIETE GENERALE','HSBC', 'STANDARD CHARTERED', 'DANSKE BANK', 'UBS'],
              index_col = 'Time Stamp')  

stats = cds_spreads_df.describe()
print(cds_spreads_df['CREDIT SUISSE'].isna().sum())

# fill missing values with interpolate

cds_spreads_df['CREDIT SUISSE'].interpolate(method = 'linear', inplace = True)
cds_spreads_df['BNP'].interpolate(method = 'linear', inplace = True)
cds_spreads_df['DEUTSCHE BANK'].interpolate(method = 'linear', inplace = True)
cds_spreads_df['SOCIETE GENERALE'].interpolate(method = 'linear', inplace = True)
cds_spreads_df['UBS'].interpolate(method = 'linear', inplace = True)

#print(cds_spreads_df['CREDIT SUISSE'].isna().sum())

#stats = cds_spreads_df.describe()


# Step 1: Take differences

cds_spreads_df['CREDIT SUISSE'].hist()

cds_spreads_df['delta CS'] = cds_spreads_df['CREDIT SUISSE'].pct_change(1)

cds_spreads_df['delta CS'].hist()

#cds_spreads_df['BNP'].interpolate(method = 'linear', inplace = True)
#cds_spreads_df['DEUTSCHE BANK'].interpolate(method = 'linear', inplace = True)
#cds_spreads_df['SOCIETE GENERALE'].interpolate(method = 'linear', inplace = True)
#cds_spreads_df['UBS'].interpolate(method = 'linear', inplace = True)