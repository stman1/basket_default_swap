# Load cds spreads from Excel sheet

library(readxl) # Excel reader package
library(dplyr) # Dataframe manipulation package
library(zoo) # Dataframe manipulation package

# read in Excel file, tab CDS_spreads_history into R dataframe
cds_basket_data=read_excel('Documents/CQF/basket_default_swap/data/default_basket_data.xlsx', 
                           'CDS_spreads_history',
                           range = 'E21:N1853')

# show cds dataframe
cds_basket_data 

# plot with gaps
plot(cds_basket_data$`Level Banco Santander`, type='o', pch=10, col='steelblue', xlab='Time Stamp', ylab='Price')

#interpolate missing values in 'Banco Santander' column
cds_basket_data <- cds_basket_data %>%
  mutate(`Level Banco Santander` = na.approx(`Level Banco Santander`))

# plot without gaps
plot(cds_basket_data$`Level Banco Santander`, type='o', pch=10, col='chocolate4', xlab='Time Stamp', ylab='Price')

# compute returns
cds_basket_data %>%
  mutate(across(everything(), .funs = funs((~(lead(.) - .) / .))))


# cds_basket_data %>%
# mutate(across('Level Credit Suisse'), .funs = (DR = list(~(lead(.) - .) / .)))


pseudo.uniform = function(X){
  Fhat <- kcde(X)
  predict(Fhat, x=X)
}


num_missing_values <-sum(is.na(cds_basket_data$'Level Credit Suisse'))
