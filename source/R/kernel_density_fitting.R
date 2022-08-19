# Load cds spreads from Excel sheet

library(readxl) # Excel reader package
library(dplyr) # Dataframe manipulation package
library(zoo) # Dataframe manipulation package
library(ks) # kernel smoothing package, alternative to MATLAB ksdensity function
library(psych) # for plotting scatterplot matrices with uniforms on the diagonal
library(here)

# Function definitions

pseudo.uniform = function(X, bw_parameter){
  Fhat <- kcde(unlist(X), h = bw_parameter)
  predict(Fhat, x=as.matrix(X))
}

empirical.cdf = function(X, bw_parameter){
  Fhat <- kcde(unlist(X), h = bw_parameter)
  return(Fhat)
}

# set path to data
data.path <- file.path(here(), '/../../data')

# read in Excel file, tab CDS_spreads_history into R dataframe
cds.spread = read_excel(file.path(data.path, 'default_basket_data.xlsx'),
                           'CDS_spreads_history',
                           range = 'E21:N1853')

# Define timestamp column as row index
cds.spread <- as.data.frame(cds.spread) # convert from tibble to data.frame
row.names(cds.spread) <- cds.spread$Timestamp
cds.spread <- cds.spread[,-1] 

# count missing values
missing_values <- cds.spread %>% #select(where(is.numeric)) %>%
  summarise_all(list(~(sum(is.na(.)))))

# Drop Banco Santander, Standard Chartered, Danske Bank: too many missing values
cds.spread <- subset(cds.spread, select = -c(`Banco Santander`, `Standard Chartered`, `Danske Bank`))

#interpolate missing values for all columns
cds.spread <- cds.spread %>%
  mutate_at(.vars = c('Credit Suisse', 'BNP', 'Deutsche Bank', 'Societe Generale', 'HSBC', 'UBS'), list(~na.approx(.)))

# show cds spread dataframe
#cds.spread 

# plot with gaps
#plot(cds.spread$`Level Banco Santander`, type='o', pch=10, col='steelblue', xlab='Time Stamp', ylab='Price')

# plot without gaps
#plot(cds.spread$`Level Banco Santander`, type='o', pch=10, col='chocolate4', xlab='Time Stamp', ylab='Price')

# compute returns
cds.returns <- cds.spread %>%
  mutate(across(is.numeric, list(ret = ~(./lead(.) - 1))))

# remove levels
cds.returns <- subset(cds.returns, select = -c(`Credit Suisse`, `BNP`, `Deutsche Bank`, `Societe Generale`, `HSBC`, `UBS`))

# Remove first data point (row) 
cds.returns <- head(cds.returns,-1)

# Restrict data frame to two years of daily data
cds.returns <- head(cds.returns, 500)

# display data frame
#  cds.spread

# scatterplot

plot(cds.returns$`Credit Suisse_ret`, cds.returns$BNP_ret)
plot(cds.returns$BNP_ret, cds.returns$`Deutsche Bank_ret`)

# scatter plot summary cds.spread
pairs.panels(cds.spread[, 1:6], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

# scatter plot summary cds.returns
pairs.panels(cds.returns[, 1:6], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

# compute empirical CDF of the CDS spread return series
fhat_credit_suisse <- empirical.cdf(unlist(cds.spread[, 8], 0.01))

# plot CDF
plot(fhat_credit_suisse, ylab="CDF")

# plot histogram
plot(histde(pseudo.uniform(unlist(cds.spread[, 8], 0.01))))









