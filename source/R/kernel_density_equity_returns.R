# Import R packages

library(readxl) # Excel reader package
library(dplyr) # Dataframe package
library(zoo) # Another Dataframe package
library(ks) # kernel smoothing package, alternative to MATLAB's ksdensity function
library(psych) # for plotting scatterplot matrices with uniforms on the diagonal
library(here) # current working directory

# Global variables

data.set = "automotive"

data.path <- switch(data.set, 
                    "retail" = file.path(here(), '/../../../CQF/Project/data/Retail'), 
                    "banks" = file.path(here(), '/../../../CQF/Project/data/Banks'), 
                    "automotive" = file.path(here(), '/../../../CQF/Project/data/Automotive'),
                    "insurance" = file.path(here(), '/../../../CQF/Project/data/Insurance'))

range.price_history <- switch(data.set,
                              "retail" = 'B2:G2821', 
                              "banks" = 'B2:J2791', 
                              "automotive" = 'B2:G2798',
                              "insurance" = 'B2:L2821')

range.entities <- switch(data.set,
                         "retail" = 'A1:B6', 
                         "banks" = 'A1:B10', 
                         "automotive" = 'A1:B6',
                         "insurance" = 'A1:B11')

num.entities <- switch(data.set,
                       "retail" =  5, 
                       "banks" = 9, 
                       "automotive" = 5,
                       "insurance" = 10)

excel.file <- 'CDS_spreads.xlsx'

sheet.price_history <- 'Equity_prices'

sheet.entities <- 'Entities'

max.datapoints <- 500 # for example: 2 years daily observations on business days is approximately 500 observations
zero.cutoff <- 0.0005 # returns / differences smaller than this threshold value are considered to be zero

# Function definitions

pseudo.uniform = function(X, bw_parameter){
  empirical_cdf <- kcde(X, h = bw_parameter)
  predict(empirical_cdf, x=X)
}

empirical.cdf = function(X, bw_parameter){
  empirical_cdf <- kcde(unlist(X), h = bw_parameter)
  return(empirical_cdf)
}

drop.name = function(series.name, spread.frame, entities.frame){
  spread.frame <- spread.frame[names(spread.frame) != series.name]
  entities.frame <- subset(entities.frame, entities.frame$Name != series.name)
  num.entities = num.entities-1
  resultList <- list("spreads" = spread.frame, "entities" = entities.frame, "count_entities" = num.entities)
  return(resultList)
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

# compute n-th day returns
frequency_n_days <- 5

equity.price <- equity.price %>%
  slice(which(row_number() %% frequency_n_days == 1))

equity.returns <- equity.price %>%
  mutate(across(where(is.numeric), list(ret = ~(log(.x/lag(.x))))))

# remove levels
equity.returns <- subset(equity.returns, select = -c(1:num.entities))

# Remove last row 
equity.returns <- tail(equity.returns,-1)

equity.returns <- equity.returns[complete.cases(equity.returns[1:num.entities]),]

# Remove all data points with zero or close to zero diffs
equity.returns <- equity.returns %>% filter(Reduce(`&`, as.data.frame(abs(.) > zero.cutoff)))

# Restrict data frame to two years of daily data
equity.returns <- tail(equity.returns, 500)

# Scatterplot

# Series 1 diffs vs. series 2 diffs
plot(equity.returns[,1], equity.returns[,4])

# Series 2 diffs vs. series 3 diffs
plot(equity.returns[,2], equity.returns[,3])

# Summary scatter plot summary cds returns
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
  # kernel density estimation using the hpi bandwidth selector
  #plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), hpi(unlist(equity.returns[, time_series_index]))), binw = user.binw))
}

time_series_index = 1
 
bw_list = c(0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625, 0.0000783125, 0.00003915625, 0.00001975, 0.000009875, 0.00000493875, 0.000002469, 0.0000012345, 0.00000000000001)

for (bw_ in bw_list){
  plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), bw = bw_), binw = user.binw), xlab=paste(cds.entities$Name[time_series_index], deparse(bw_)))
  # kernel density estimation using the hpi bandwidth selector
  #plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), hpi(unlist(equity.returns[, time_series_index]))), binw = user.binw))
}


bmw.uniform <- histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), bw = bw_), binw = user.binw)$

time_series_index = 1
user.bw = 0.0000625 #  0.000099
user.binw = 0.09 # 0.09
plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), bw = user.bw), binw = user.binw), xlab=paste(cds.entities$Name[time_series_index], deparse(user.bw)))











