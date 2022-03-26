# Title: AMELIA II Imputation

# Notes:
	#* Description: Script to do MI with AMELIA and download those resulting data

# Setup
	#* Load packages
library(Amelia)
library(dplyr)
library(haven)
	#* Load Data
sim_three = read.csv('data/sim_three_MAR.csv')
sim_ten = read.csv('data/sim_ten_MAR.csv')
anes = read.csv('data/anes_2020_clean.csv')

# Set seed
set.seed(1234)
# Perform imputation with AMELIA
	#* Simulated
sim_three_imp = amelia(sim_three, m = 5)


sim_ten_imp = amelia(sim_ten, m = 5)


	#* ANES
anes_imp = amelia(anes, m = 5, p2s = 0, parallel = 'multicore')


# Save datasets ----
saveRDS(sim_three_imp, 'data/amelia_sim_3.RDS')
write.csv(sim_three_imp$imputations[[5]], 'data/amelia_sim_three_partial.csv')
saveRDS(sim_ten_imp, 'data/amelia_sim_10.RDS')
write.csv(sim_ten_imp$imputations[[5]], 'data/amelia_sim_ten_partial.csv')
saveRDS(anes_imp, 'data/amelia_anes.RDS')
write.csv(anes_imp$imputations[[5]], 'data/amelia_anes_partial.csv')
save.image('data/amelia_image.RData')