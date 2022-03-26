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
import re
from tabulate import tabulate
import os

	#* Plot Style 
sns.set(style = 'whitegrid')
	#* Working directory 
#os.chdir('/home/damoncroberts/Dropbox/current_projects/dcr_rf_imputation/') # Linux
os.chdir('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/') # Mac
	#* Import ANES data
anes = pd.read_csv('data/anes_2020_clean.csv', low_memory = False)

	#* Clean ANES data
anes.columns.get_loc('Unnamed: 0')
anes = anes.drop(['Unnamed: 0'], axis = 1)

# Define imputation class
class anes_impute:
	def __init__(self, ndata = 5, niter = 7, nestimators = 30):
		self.anes_kernel = mf.ImputationKernel(anes, 
			datasets = ndata, 
			save_all_iterations=True, 
			random_state=601)
		self.anes_kernel.mice(iterations = niter, n_estimators = nestimators)
		for i in range(0, 4):
			self.anes_data = self.anes_kernel.complete_data(dataset = i, inplace = False)
			pd.DataFrame.to_csv(self.anes_data, 
				path_or_buf = 'data/anes_rf_imputed_' + str(i) + '.csv')
anes_impute()