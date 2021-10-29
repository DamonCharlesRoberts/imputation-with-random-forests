# Title: Data Characterization

# Notes:
	#* Description: Looking at multivariate distributions of simulated and generated data
	#* Updated: 2021-10-28
	#* Updated by: dcr
# Setup
	#* Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
	#* Load in datasets
sim1 = pd.read_csv('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary.csv')
sim1_sparse = pd.read_csv('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary_sparse.csv')
sim10 = pd.read_csv('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim.csv')
sim10_sparse = pd.read_csv('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim_sparse.csv')
	#* Figure aesthetics
sns.set(style ='ticks')

# Figures
	#* Bivariate Simulated
bi_sim = sns.kdeplot(sim1['Column_A'], shade = True, color = 'grey').set(xlabel=None)
bi_sim2 = sns.kdeplot(sim1['Column_B'], shade = True, color = 'black').set(xlabel=None)
bi_plot = plt.legend(labels = ['Column A', 'Column B'])
plt.savefig('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/figures/sim_bivariate_distplot.png')