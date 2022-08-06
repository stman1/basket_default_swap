# Load cds spreads from Excel sheet

library(readxl) # Excel reader package
library(dplyr) # Dataframe manipulation package
library(zoo) # Dataframe manipulation package
library(ks) # kernel smoothing package, alternative to MATLAB ksdensity function
library(psych) # to plot nice scatterplot matrices
library(here)

# set path to data

data.path <- file.path(here(), '/../../data')
# read in Excel file, tab CDS_spreads_history into R dataframe
cds_spread_data=read_excel(file.path(data.path, 'default_basket_data.xlsx'),
                           'CDS_spreads_history',
                           range = 'E21:N1853')

# count missing values
missing_values <- cds_spread_data %>% select(where(is.numeric)) %>%
  summarise_all(list(~(sum(is.na(.)))))

# Drop Banco Santander, Standard Chartered, Danske Bank: too many missing values
cds_spread_data <- subset(cds_spread_data, select = -c(`Banco Santander`, `Standard Chartered`, `Danske Bank`))

#interpolate missing values for all columns
cds_spread_data <- cds_spread_data %>%
  mutate_at(.vars = c('Credit Suisse', 'BNP', 'Deutsche Bank', 'Societe Generale', 'HSBC', 'UBS'), list(~na.approx(.)))

# show cds dataframe
#cds_spread_data 

# plot with gaps
#plot(cds_spread_data$`Level Banco Santander`, type='o', pch=10, col='steelblue', xlab='Time Stamp', ylab='Price')

# plot without gaps
#plot(cds_spread_data$`Level Banco Santander`, type='o', pch=10, col='chocolate4', xlab='Time Stamp', ylab='Price')

# compute returns
cds_spread_data <- cds_spread_data %>%
  mutate(across(is.numeric, list(ret = ~(./lead(.) - 1))))

# Remove first data point (row) 
cds_spread_data <- head(cds_spread_data,-1)

# display data frame
cds_spread_data



# scatter plot

pairs.panels(cds_spread_data[,8:12], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)


# compute empirical CDF of the CDS spread return series

#fhat_credit_suisse <- pseudo.uniform(unlist(cds_spread_data[, 2]))

fhat_credit_suisse <- kcde(unlist(cds_spread_data[, 8]))
predict(fhat_credit_suisse, x=as.matrix(cds_spread_data[, 8]))


plot(fhat_credit_suisse, ylab="CDF")

histde(fhat_credit_suisse)
plot(histde(fhat_credit_suisse))

# Functions
pseudo.uniform = function(X){
  Fhat <- kcde(X)
  predict(Fhat, x=as.matrix(X))
}







