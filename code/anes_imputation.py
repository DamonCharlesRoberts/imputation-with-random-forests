# Title: ANES Imputation ----

# Notes: ----
    #* Description: Script to clean ANES and perform imputation on missing values in it ----
    #* Updated: 2022-03-21 ----
    #* Updated by: dcr ----

# Set up ----
    #*  Load functions and modules ----
import miceforest as mf
import pandas as pd
import numpy as np
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
	#* Source rmse function file
exec(open('code/rmse_function.py').read())
    #* Load 2020 ANES ----
anes = pd.read_csv('data/anes_timeseries_2020.csv')

# Recode NA's ----
anes = anes.drop(columns = ['V200001', 'V160001_orig', 'V200002', 'V200003', 'V200004', 'V200005', 'V200006', 'V200007', 'V200008', 'V200009', 'V200010a', 'V200010b', 'V200010c', 'V200010d', 'V200011a', 'V200011b', 'V200011c', 'V200011d', 'V200012a', 'V200012b', 'V200012c', 'V200012d', 'V200013a', 'V200013b', 'V200013c', 'V200013d', 'V200014a', 'V200014b', 'V200014c', 'V200014d', 'V200015a', 'V200015b', 'V200015c', 'V200015d', 'V200016a', 'V200016b', 'V200016c', 'V200016d', 'V201001', 'V201002a', 'V201002b', 'V201003', 'V201004'])._get_numeric_data()
anes[anes<0] = pd.NA #make anyvalue less than 0 an NA value


# Imputation
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
				path_or_buf = '/home/damoncroberts/Dropbox/current_projects/dcr_rf_imputation/data/anes_impute_' + str(i) + '.csv')
anes_impute()

anes_kernel = mf.ImputationKernel(
                                   anes,
                                   datasets=1,
                                   save_all_iterations=True,
                                   random_state=601)
anes_kernel.mice(10)