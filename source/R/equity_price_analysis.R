# Load equity prices from Excel sheet

library(readxl) # Excel reader package
library(dplyr) # Dataframe manipulation package
library(zoo) # Dataframe manipulation package
library(ks) # kernel smoothing package, alternative to MATLAB ksdensity function
library(psych) # to plot nice scatterplot matrices with uniforms on the diagonal
library(here)

# set path to data
data.path <- file.path(here(), '/../../data')

# read in Excel file, tab Equity_prices into R dataframe
equity.price = read_excel(file.path(data.path, 'default_basket_data.xlsx'),
                           'Equity_prices',
                           range = 'B3:K1912')

# Define timestamp column as row index
equity.price <- as.data.frame(equity.price)
row.names(equity.price) <- equity.price$Timestamp
equity.price <- equity.price[,-1] 

# count missing values
missing_values <- equity.price %>% #select(where(is.numeric)) %>%
  summarise_all(list(~(sum(is.na(.)))))

# Drop Credit Suisse, UBS
equity.price <- subset(equity.price, select = -c(`Credit Suisse`, `UBS`))

#interpolate missing values for all columns
equity.price <- equity.price %>%
  mutate_at(.vars = c('Santander', 'BNP', 'Deutsche Bank', 'Danske Bank', 'Societe Generale', 'Standard Chartered'), list(~na.approx(.)))

# show equity price dataframe
equity.price 

# compute returns
equity.returns <- equity.price %>%
  transmute(across(is.numeric, list(ret = ~(./lead(.) - 1))))

# Remove first data point (row) 
equity.price <- head(equity.price,-1)

# Restrict data frame to two years of daily data
equity.price <- equity.price[equity.price$Timestamp > '2020-06-30',]

# display data frame
equity.price

# scatterplot

plot(equity.price$`Credit Suisse_ret`, equity.price$BNP_ret)
plot(equity.price$BNP_ret, equity.price$`Deutsche Bank_ret`)

# scatter plot summary
pairs.panels(equity.price[, 8:12], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)


# compute empirical CDF of the CDS spread return series
fhat_credit_suisse <- empirical.cdf(unlist(equity.price[, 8], 0.01))

# plot CDF
plot(fhat_credit_suisse, ylab="CDF")

# plot histogram
plot(histde(pseudo.uniform(unlist(equity.price[, 8], 0.01))))




# Functions
pseudo.uniform = function(X, bw_parameter){
  Fhat <- kcde(unlist(X), h = bw_parameter)
  predict(Fhat, x=as.matrix(X))
}

empirical.cdf = function(X, bw_parameter){
  Fhat <- kcde(unlist(X), h = bw_parameter)
  return(Fhat)
}







