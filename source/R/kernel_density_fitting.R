# Import R packages

library(readxl) # Excel reader package
library(dplyr) # Dataframe package
library(zoo) # Another Dataframe package
library(ks) # kernel smoothing package, alternative to MATLAB's ksdensity function
library(psych) # for plotting scatterplot matrices with uniforms on the diagonal
library(here) # current working directory

# Global variable values

data.set = "insurance"

data.path <- switch(data.set, 
                    "retail" = file.path(dirname(dirname(dirname(getwd()))),'basket_default_swap/data/Retail'), 
                    "banks" = file.path(dirname(dirname(dirname(getwd()))),'basket_default_swap/data/Banks'), 
                    "automotive" = file.path(dirname(dirname(dirname(getwd()))),'basket_default_swap/data/Automotive'),
                    "insurance" = file.path(dirname(dirname(dirname(getwd()))),'basket_default_swap/data/Insurance'))

range.spreads_history <- switch(data.set,
                     "retail" = 'E21:J2104', 
                     "banks" = 'E21:N1975', 
                     "automotive" = 'E21:J2104',
                     "insurance" = 'E21:O1974')

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

sheet.spreads_history <- 'CDS_spreads_history'

sheet.entities <- 'Entities'

max.datapoints <- 500 # for example: 2 years daily observations on business days is approximately 500 observations

zero.cutoff <- 0.01 # spread diffs smaller than this threshold value are considered to be zero

# Function definitions

pseudo.uniform = function(X, bw_parameter){
  empirical_cdf <- kcde(unlist(X), h = bw_parameter)
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

# load cds spreads time series from Excel sheet CDS_spreads_history into R dataframe
cds.spread = read_excel(file.path(data.path, excel.file),
                           sheet.spreads_history,
                           range = range.spreads_history)

# load entity names from Excel sheet 'Entities'  
cds.entities = read_excel(file.path(data.path, excel.file),
                        sheet.entities,
                        range = range.entities)

# Make the date column the data.frame row index
cds.spread <- as.data.frame(cds.spread) # convert from tibble to object of type data.frame
row.names(cds.spread) <- cds.spread$Timestamp
cds.spread <- cds.spread[,-1] # remove index column, dataframe is now indexed by date

# count missing values
missing_values <- cds.spread %>% #select(where(is.numeric)) %>%
  summarise_all(list(~(sum(is.na(.)))))

# Remove all rows with NAs
cds.spread <- cds.spread[complete.cases(cds.spread[1:num.entities]),]


#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#------------Individual data adjustments - change for each data set----------------------------------#

# Drop Generali

mutated_frames = drop.name('Generali', cds.spread, cds.entities)
cds.spread <- mutated_frames$spreads
cds.entities <- mutated_frames$entities
num.entities <- mutated_frames$count_entities
rm(mutated_frames)


#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#-------- End of individual data adjustments - change for each data set------------------------------#

# Delete all rows with at least one NA
cds.spread <- cds.spread[complete.cases(cds.spread[1:num.entities]),]

# compute n-th day returns
frequency_n_days <- 5

cds.spread <- cds.spread %>%
  slice(which(row_number() %% frequency_n_days == 1))


# Linear interpolation of missing values for all columns
cds.spread <- cds.spread %>%
  mutate_at(.vars = colnames(cds.spread), list(~na.approx(.)))



# Diffs, Returns, Log returns

# 1. Diffs

cds.diffs <- as.data.frame(lapply(cds.spread, diff, lag=1))
rm(cds.spread)

# Remove all rows with same diffs
cds.diffs <- cds.diffs %>% filter(colnames(cds.diffs)[1] != colnames(cds.diffs)[2])
#cds.diffs <- cds.diffs %>% filter(cds.diffs$Aegon != cds.diffs$Allianz)

# Remove all data points with zero or close to zero diffs
cds.diffs <- cds.diffs %>% filter(Reduce(`&`, as.data.frame(abs(.) > zero.cutoff)))

# Restrict data frame to two years of daily data
cds.diffs <- head(cds.diffs, max.datapoints)

# Scatterplot

# Series 1 diffs vs. series 2 diffs
plot(cds.diffs[,1], cds.diffs[,2])

# Series 2 diffs vs. series 3 diffs
plot(cds.diffs[,2], cds.diffs[,3])


# Summary scatter plot summary cds returns
# Observation: shows uni-modal distribution
pairs.panels(cds.diffs[, 1:num.entities], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)


# compute empirical CDF of the CDS spread return series
empirical_cdf <- empirical.cdf(unlist(cds.diffs[, time_series_index], 0.01))

# plot CDF
plot(empirical_cdf, ylab="CDF")


# plot histogram
user.bw = 0.0005 # 0.0003
user.binw = 0.09 # 0.09

for (time_series_index in 1:num.entities){
  plot(histde(pseudo.uniform(unlist(cds.diffs[, time_series_index]), bw = user.bw), binw = user.binw), xlab=paste(cds.entities$Name[time_series_index], deparse(user.binw)))
  # kernel density estimation using the hpi bandwidth selector
  #plot(histde(pseudo.uniform(unlist(cds.diffs[, time_series_index]), hpi(unlist(cds.diffs[, time_series_index]))), binw = user.binw))
}

time_series_index = 3

bw_list = c(0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625, 0.0000783125, 0.00003915625, 0.00001975, 0.000009875, 0.00000493875, 0.000002469, 0.0000012345, 0.00000000000001)

for (bw_ in bw_list){
  plot(histde(pseudo.uniform(unlist(cds.diffs[, time_series_index]), bw = bw_), binw = user.binw), xlab=paste(cds.entities$Name[time_series_index], deparse(bw_)))
  # kernel density estimation using the hpi bandwidth selector
  #plot(histde(pseudo.uniform(unlist(cds.diffs[, time_series_index]), hpi(unlist(cds.diffs[, time_series_index]))), binw = user.binw))
}


user.bw = 0.0000625 #  0.000099
user.binw = 0.09 # 0.09
plot(histde(pseudo.uniform(unlist(cds.diffs[, time_series_index]), bw = user.bw), binw = user.binw), xlab=paste(cds.entities$Name[time_series_index], deparse(user.bw)))


     
     
     











