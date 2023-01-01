# Import R packages

library(readxl) # Excel reader package
library(dplyr) # Dataframe package
library(zoo) # Another Dataframe package
library(ks) # kernel smoothing package, alternative to MATLAB's ksdensity function
library(psych) # for plotting scatterplot matrices with uniforms on the diagonal
library(here) # current working directory

# Global variable values

data.set = "retail"

data.path <- switch(data.set, 
                    "retail" = file.path(here(), '/../../../CQF/Project/data/Retail'), 
                    "banks" = file.path(here(), '/../../../CQF/Project/data/Banks'), 
                    "automotive" = file.path(here(), '/../../../CQF/Project/data/Automotive'))

range.spreads_history <- switch(data.set,
                     "retail" = 'E21:J2104', 
                     "banks" = 'E21:N1975', 
                     "automotive" = 'E21:J2104')

range.entities <- switch(data.set,
                         "retail" = 'A1:B6', 
                         "banks" = 'A1:B10', 
                         "automotive" = 'A1:B6')


num.entities <- switch(data.set,
                       "retail" =  5, 
                       "banks" = 9, 
                       "automotive" = 5)

excel.file <- 'CDS_spreads.xlsx'

sheet.spreads_history <- 'CDS_spreads_history'

sheet.entities <- 'Entities'

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


#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#------------Individual data adjustments - change for each data set----------------------------------#

# Remove first 70 data points (Ahold data missing) 
cds.spread <- tail(cds.spread,-70)

# Linear interpolation of missing values for all columns

cds.spread <- cds.spread %>%
  mutate_at(.vars = colnames(cds.spread), list(~na.approx(.)))

# compute n-th day returns
frequency_n_days <- 5

cds.spread <- cds.spread %>%
  slice(which(row_number() %% frequency_n_days == 1))

cds.diffs <- cds.spread %>%
  mutate(across(is.numeric, list(diff = ~(. - lead(.)))))

# remove levels
cds.diffs <- subset(cds.diffs, select = -c(1:num.entities))


# Remove first data point (row) 
cds.diffs <- head(cds.diffs,-1)

# Remove all data points with zero or close to zero diffs
cds.diffs <- cds.diffs %>% filter(Reduce(`&`, as.data.frame(abs(.) > zero.cutoff)))

# Restrict data frame to two years of daily data
cds.diffs <- head(cds.diffs, max.datapoints)

# Scatterplot

# Ahold vs. Carrefour returns
plot(cds.diffs$`Ahold_Delhaize_ret`, cds.diffs$Carrefour_ret)

# Kering vs. Next UK returns
plot(cds.diffs$Kering_ret, cds.diffs$`Next_UK_ret`)

# Summary scatter plot cds spreads (level)
# Observation: clearly shows bi-modal distribution typical of "level" data
pairs.panels(cds.spread[, 1:6], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

# Summary scatter plot summary cds returns
# Observation: shows uni-modal distribution
pairs.panels(cds.diffs[, 1:5], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

# 1: Ahold_Delhaize, 2: Carrefour, 3: Kering, 4: Next UK, 5: Tesco
time_series_index <- 1

# compute empirical CDF of the CDS spread return series
empirical_cdf <- empirical.cdf(unlist(cds.diffs[, time_series_index], 0.01))

# plot CDF
plot(empirical_cdf, ylab="CDF")

# plot histogram
plot(histde(pseudo.uniform(unlist(cds.diffs[, time_series_index]), bw = 0.0008), binw = 0.03))
plot(histde(pseudo.uniform(unlist(cds.diffs[, time_series_index]), hpi(unlist(cds.diffs[, time_series_index]))), binw = 0.09))


     
     
     
     
     











