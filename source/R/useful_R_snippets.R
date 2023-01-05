

# Drop UBS 
equity.price <- subset(equity.price, select = -c(`UBS`))

# show equity.price dataframe

equity.price 

# plot with gaps
plot(equity.price$`Level Banco Santander`, type='o', pch=10, col='steelblue', xlab='Time Stamp', ylab='Price')

# plot without gaps
plot(equity.price$`Level Banco Santander`, type='o', pch=10, col='chocolate4', xlab='Time Stamp', ylab='Price')


Ahold

threshold_near_zero = 0.001

plot(histde(pseudo.uniform(unlist(equity.returns[, time_series_index]), bw = 0.00015), binw = 0.095))

# Remove first 70 data points (Ahold data missing) 

cds.spread <- tail(cds.spread,-70)

data.path <- file.path(here(), '/../../../CQF/Project/data/Retail')

cds.spread <- cds.spread %>%
  mutate_at(.vars = c('Ahold_Delhaize', 'Carrefour', 'Kering', 'Next_UK', 'Tesco'), list(~na.approx(.)))

cds.diffs <- subset(cds.diffs, select = -c(`Ahold_Delhaize`, `Carrefour`, `Kering`, `Next_UK`, `Tesco`))


# Summary scatter plot cds spreads (level)
# Observation: clearly shows bi-modal distribution typical of "level" data
pairs.panels(cds.spread[, 1:6], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)


# clumsy way of computing diffs
cds.diffs <- cds.spread %>%
  mutate(across(where(is.numeric), list(diff = ~(. - lead(.)))))

# count missing values
missing_values <- cds.spread %>% #select(where(is.numeric)) %>%
  summarise_all(list(~(sum(is.na(.)))))

# 2. Log Returns

cds.diffs <- cds.spread %>%
  mutate(across(where(is.numeric), list(diff = ~(log(./lead(.))))))
# remove levels
cds.diffs <- subset(cds.diffs, select = -c(1:num.entities))
# Remove first data point (row) 
cds.diffs <- head(cds.diffs,-1)


# Remove first 70 data points (Ahold data missing) 
cds.spread <- tail(cds.spread,-461)

cds.spread <- head(cds.spread,-460)
