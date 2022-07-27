# Load cds spreads from Excel sheet

library(readxl) # Excel readerb package
library(dplyr) # Dataframe manipulation package
library(zoo) # Dataframe manipulation package



cds_basket_data=read_excel('Documents/CQF/basket_default_swap/data/default_basket_data.xlsx', 
                           'CDS_spreads_history',
                           range = 'E21:N1853')
cds_basket_data # show cds dataframe

plot(cds_basket_data$`Banco Santander`, type='o', pch=10, col='steelblue', xlab='Time Stamp', ylab='Price')

#interpolate missing values in 'Banco Santander' column
cds_basket_data <- cds_basket_data %>%
  mutate(`Banco Santander` = na.approx(`Banco Santander`))


pseudo.uniform = function(X){
  Fhat <- kcde(X)
  predict(Fhat, x=X)
}


num_missing_values <-sum(is.na(cds_basket_data$'Credit Suisse'))
