# Import R packages

library(readxl) # Excel reader package
library(dplyr) # Dataframe package
library(zoo) # Another Dataframe package
library(ks) # kernel smoothing package, alternative to MATLAB's ksdensity function
library(psych) # for plotting scatterplot matrices with uniforms on the diagonal
library(here) # current working directory

# Global variables

data.set = "retail"

data.path <- switch(data.set, 
                    "retail" = file.path(here(), '/../../../CQF/Project/data/Retail'), 
                    "banks" = file.path(here(), '/../../../CQF/Project/data/Banks', 
                                        "automotive" = file.path(here(), '/../../../CQF/Project/data/Automotive')))

max.datapoints <- 500 # for example: 2 years daily observations on business days is approximately 500 observations
zero.cutoff <- 0.001 # returns / differences smaller than this threshold value are considered to be zero

# Function definitions

pseudo.uniform = function(X, bw_parameter){
  empirical_cdf <- kcde(unlist(X), h = bw_parameter)
  predict(empirical_cdf, x=X)
}

empirical.cdf = function(X, bw_parameter){
  empirical_cdf <- kcde(unlist(X), h = bw_parameter)
  return(empirical_cdf)
}

# read in Excel file, tab CDS_spreads_history into R dataframe
equity.price = read_excel(file.path(data.path, 'CDS_spreads.xlsx'),
                        'Equity_prices',
                        range = 'B2:G2821')

# Define timestamp column as row index
equity.price <- as.data.frame(equity.price) # convert from tibble to data.frame
row.names(equity.price) <- equity.price$Timestamp
equity.price <- equity.price[,-1] 

# count missing values
missing_values <- equity.price %>% #select(where(is.numeric)) %>%
  summarise_all(list(~(sum(is.na(.)))))

# Drop first row (missing values)
equity.price <- equity.price[-1, ]

# interpolate missing values (all columns)

equity.price <- equity.price %>%
  mutate_at(.vars = c('Ahold_Delhaize', 'Carrefour', 'Kering', 'Next_UK', 'Tesco'), list(~na.approx(.)))

# compute n-th day returns
frequency_n_days <- 1

equity.price <- equity.price %>%
  slice(which(row_number() %% frequency_n_days == 1))

equity.returns <- equity.price %>%
  mutate(across(is.numeric, list(ret = ~(log(./lead(.))))))

# remove levels
equity.returns <- subset(equity.returns, select = -c(`Ahold_Delhaize`, `Carrefour`, `Kering`, `Next_UK`, `Tesco`))

# Remove first data point (row) 
equity.returns <- head(equity.returns,-1)

# Remove all data points with zero or close to zero diffs

threshold_near_zero = 0.0001

equity.returns <- equity.returns %>% filter(Reduce(`&`, as.data.frame(abs(.) > threshold_near_zero)))

# Restrict data frame to two years of daily data
equity.returns <- tail(equity.returns, 500)

# display data frame
#  equity.price

# scatterplot

# Credit Suisse returns vs. BNP returns
plot(equity.returns$`Ahold_Delhaize_ret`, equity.returns$Carrefour_ret)

# BNP vs. Deutsche Bank
plot(equity.returns$Ahold_Delhaize_ret, equity.returns$`Kering_ret`)

# Summary scatter plot cds spreads (level)
# clearly shows bi-modal distribution typical of "level" data
pairs.panels(equity.price[, 1:6], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

# Summary scatter plot summary cds returns
# Observations: shows uni-modal distribution
pairs.panels(equity.returns[, 1:6], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

# 1: Credit Suisse, 2: BNP, 3: Deutsche Bank, 4: Societe Generale, 5: HSBC, 6: UBS
time_series_index <- 5

# compute empirical CDF of the CDS spread return series
empirical_cdf <- empirical.cdf(unlist(equity.returns[, time_series_index], 0.01))

# plot CDF
plot(empirical_cdf, ylab="CDF")

# plot histogram

plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), bw = 0.0001), binw = 0.095))

plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), hpi(unlist(equity.returns[, time_series_index]))), binw = 0.08))
















