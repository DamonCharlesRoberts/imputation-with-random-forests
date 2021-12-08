# Title: AMELIA II Imputation

# Notes:
	#* Description: Script to do MI with AMELIA and download those resulting data

# Setup
	#* Load packages
library(Amelia)
library(dplyr)
	#* Load Data
sim_three = read.csv('data/sim_three_MAR.csv')
sim_ten = read.csv('data/sim_ten_MAR.csv')
wvs = read.csv('data/wvs_original.csv') %>%
	filter(A_YEAR == 2018) %>%
	select(-c(A_YEAR, Q82_ECO, Q82_NAFTA, version, doi, B_COUNTRY_ALPHA, C_COW_ALPHA, LNGE_ISO, Partyname, Partyabb, CPARTY, CPARTYABB, A_WAVE, A_STUDY))

# Perform imputation with AMELIA
	#* Simulated
sim_three_imp = amelia(sim_three, m = 4)
sim_three_imp_one = sim_three_imp[4]

sim_ten_imp = amelia(sim_ten, m = 4)
sim_ten_imp_one = sim_ten_imp[4]

	#* WVS
wvs_imp = amelia(wvs, m = 4)
