# Title: DCR Random Forest Imputation Primary Analysis

# Notes:
	#* Description: Performs primary analyses examining the performance of RF implementation of MICE to other missingness approaches
	#* Updated: 2021-11-12
	#* Updated by: dcr
# Setup:
	#* Libraries
import miceforest as mf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_density
	#* Plot Style 
sns.set(style = 'whitegrid')
# Simulated Data 
	#* Import complete data
sim_3_comp = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_complete.csv')
sim_10_comp = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_complete.csv')
	#* Import MAR data
sim_3_mar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_MAR.csv')
sim_10_mar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_MAR.csv')
	#* Import MCAR data
sim_3_mcar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_MCAR.csv')
sim_10_mcar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_MCAR.csv')
	#* Import MNAR data
sim_3_mnar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_MNAR.csv')
sim_10_mnar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_MNAR.csv')

# 3 variable

