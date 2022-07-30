# Load cds spreads from Excel sheet

library(readxl) # Excel reader package
library(dplyr) # Dataframe manipulation package
library(zoo) # Dataframe manipulation package
library(ks) # kernel smoothing package, alternative to MATLAB ksdensity


# read in Excel file, tab CDS_spreads_history into R dataframe
cds_spread_data=read_excel('C:/CQF/Project/data/default_basket_data.xlsx', 
                           'CDS_spreads_history',
                           range = 'E21:N1853')

# count missing values
missing_values <- cds_spread_data %>% select(is.numeric) %>%
  summarise_all(list(~(sum(is.na(.)))))

# Drop Banco Santander, Standard Chartered, Danske Bank
cds_spread_data <- subset(cds_spread_data, select = -c(`Banco Santander`, `Standard Chartered`, `Danske Bank`))

#interpolate missing values for all columns
#cds_spread_data <- cds_spread_data %>%
#  mutate(`Level Banco Santander` = na.approx(`Level Banco Santander`))

cds_spread_data <- cds_spread_data %>%
  mutate(across(is.numeric, list(~na.approx(.))))

# show cds dataframe
#cds_basket_data 

# plot with gaps
#plot(cds_spread_data$`Level Banco Santander`, type='o', pch=10, col='steelblue', xlab='Time Stamp', ylab='Price')

# plot without gaps
#plot(cds_spread_data$`Level Banco Santander`, type='o', pch=10, col='chocolate4', xlab='Time Stamp', ylab='Price')

# compute returns

cds_spread_data <- cds_spread_data %>%
  mutate(across(is.numeric, list(perc_change = ~(./lag(.) - 1))))

cds_spread_data

num_missing_values <-sum(is.na(cds_spread_data$'Level Credit Suisse'))

pseudo.uniform = function(X){
  Fhat <- kcde(X)
  predict(Fhat, x=X)
}







