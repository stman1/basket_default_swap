# Import R packages

library(readxl) # Excel reader package
library(dplyr) # Dataframe package
library(zoo) # Another Dataframe package
library(ks) # kernel smoothing package, alternative to MATLAB's ksdensity function
library(psych) # for plotting scatterplot matrices with uniforms on the diagonal
library(here) # current working directory

# Global variables

data.set = "final_basket"

data.path <- file.path(dirname(dirname(dirname(getwd()))), 'basket_default_swap/data/final_basket')

range.price_history <- 'B2:G2821' 
                              
range.entities <- 'A1:B6'

num.entities <- 5

excel.file <- 'CDS_spreads_basket.xlsx'

sheet.price_history <- 'Equity_prices'

sheet.entities <- 'Entities'

max.datapoints <- 500 # for example: 2 years daily observations on business days is approximately 500 observations

# Function definitions

pseudo.uniform = function(X, bw_parameter){
  empirical_cdf <- kcde(X, h = bw_parameter)
  predict(empirical_cdf, x=X)
}

empirical.cdf = function(X, bw_parameter){
  empirical_cdf <- kcde(unlist(X), h = bw_parameter)
  return(empirical_cdf)
}

# read in Excel file, tab CDS_spreads_history into R dataframe
equity.price = read_excel(file.path(data.path, excel.file),
                          sheet.price_history,
                          range = range.price_history)

# load entity names from Excel sheet 'Entities'  
cds.entities = read_excel(file.path(data.path, excel.file),
                          sheet.entities,
                          range = range.entities)

# Make the date column the data.frame row index
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
  mutate_at(.vars = colnames(equity.price), list(~na.approx(.)))

equity.returns <- equity.price %>%
  mutate(across(where(is.numeric), list(ret = ~(log(.x/lag(.x))))))

# remove levels
equity.returns <- subset(equity.returns, select = -c(1:num.entities))

equity.returns <- equity.returns[complete.cases(equity.returns[1:num.entities]),]

# Restrict data frame to two years of daily data
equity.returns <- tail(equity.returns, max.datapoints)

# Scatterplot

# Series 1 diffs vs. series 2 diffs
plot(equity.returns[,1], equity.returns[,4])

# Series 2 diffs vs. series 3 diffs
plot(equity.returns[,2], equity.returns[,3])

# Summary scatter plot for equity returns data
# Observations: shows uni-modal distribution
pairs.panels(equity.returns[, 1:num.entities], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

time_series_index = 1
# compute empirical CDF of the CDS spread return series
empirical_cdf <- empirical.cdf(unlist(equity.returns[, time_series_index], 0.01))

# plot CDF
plot(empirical_cdf, ylab="CDF")

# plot histogram
user.bw = 0.0005 # 0.0003
user.binw = 0.09 # 0.09

for (time_series_index in 1:num.entities){
  plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), bw = user.bw), binw = user.binw), xlab=paste(cds.entities$Name[time_series_index], deparse(user.binw)))
}

time_series_index = 5

bw_list = c(0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625, 0.0000783125, 0.00003915625, 0.00001975, 0.000009875, 0.00000493875, 0.000002469, 0.0000012345, 0.00000000000001)

for (bw_ in bw_list){
  plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), bw = bw_), binw = user.binw), xlab=paste(cds.entities$Name[time_series_index], deparse(bw_)))
}

#         Optimal bandwidth selection per name by visual inspection of histograms

# Prudential 0.0003125
# BMW 0.0003125
# VW 0.0000783125
# Deutsche Bank 0.00001975
# Kering 0.00015625

prudential.uniform <- histde(pseudo.uniform(unlist(equity.returns[, 1]), bw = 0.0003125), binw = 0.09)
bmw.uniform <- histde(pseudo.uniform(unlist(equity.returns[, 2]), bw = 0.0003125), binw = 0.09)
vw.uniform <- histde(pseudo.uniform(unlist(equity.returns[, 3]), bw = 0.0000783125), binw = 0.09)
deutschebank.uniform <- histde(pseudo.uniform(unlist(equity.returns[, 4]), bw = 0.00001975), binw = 0.09)
kering.uniform <- histde(pseudo.uniform(unlist(equity.returns[, 5]), bw = 0.00015625), binw = 0.09)

# Define data frame for the uniform pseudo samples

pseudo.samples <- data.frame(col1 = prudential.uniform$x, 
                             col2 = bmw.uniform$x,
                             col3 = vw.uniform$x,
                             col4 = deutschebank.uniform$x,
                             col5 = kering.uniform$x)

colnames(pseudo.samples) <- c(cds.entities$Name)


# change path to output directory to data/PseudoSamples for saving 
setwd(file.path(dirname(dirname(here())), '/data/PseudoSamples'))

# save dataframe for import into python
write.csv (x = as.data.frame(pseudo.samples), file = "pseudo.samples.csv", sep=",")






