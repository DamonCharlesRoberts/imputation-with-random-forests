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
os.chdir('/home/damoncroberts/Dropbox/current_projects/dcr_rf_imputation/') # Linux
#os.chdir('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/') # Mac
	#* Import WVS data
wvs = pd.read_csv('data/wvs_original.csv', low_memory = False)
# Clean WVS

wvs = wvs[wvs['A_YEAR'] == 2018].drop(columns = ['version', 'doi', 'A_WAVE', 'A_STUDY', 'B_COUNTRY', 'B_COUNTRY_ALPHA', 'C_COW_NUM', 'C_COW_ALPHA', 'A_YEAR', 'D_INTERVIEW', 'J_INTDATE', 'FW_END', 'FW_START', 'K_TIME_START', 'K_TIME_END', 'K_DURATION', 'Q_MODE', 'N_REGION_ISO', 'N_REGION_WVS', 'N_TOWN', 'O1_LONGITUDE', 'O2_LATITUDE', 'S_INTLANGUAGE', 'LNGE_ISO', 'E_RESPINT', 'F_INTPRIVACY', 'E1_LITERACY', 'W_WEIGHT', 'S018', 'PWGHT', 'S025', 'Partyname', 'Partyabb', 'CPARTY', 'CPARTYABB']).rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

class wvs_impute:
	def __init__(self, ndata = 5, niter = 7, nestimators = 30):
		self.wvs_kernel = mf.ImputationKernel(wvs, 
			datasets = ndata, 
			save_all_iterations=True, 
			random_state=601)
		self.wvs_kernel.mice(iterations = niter, n_estimators = nestimators)
		for i in range(0, 4):
			self.wvs_data = self.wvs_kernel.complete_data(dataset = i, inplace = False)
			pd.DataFrame.to_csv(self.wvs_data, 
				path_or_buf = '/home/damoncroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_impute_' + str(i) + '.csv')
wvs_impute()



#wvs_mice_comp_1 = wvs_kernel.complete_data(dataset=0, inplace=False)
#pd.DataFrame.to_csv(wvs_mice_comp_1, path_or_buf = '/data/wvs_impute_1.csv')
#wvs_mice_comp_2 = wvs_kernel.complete_data(dataset=1, inplace=False)
#pd.DataFrame.to_csv(wvs_mice_comp_2, path_or_buf = '/data/wvs_impute_2.csv')
#wvs_mice_comp_3 = wvs_kernel.complete_data(dataset=2, inplace=False)
#pd.DataFrame.to_csv(wvs_mice_comp_3, path_or_buf = '/data/wvs_impute_3.csv')
#wvs_mice_comp_4 = wvs_kernel.complete_data(dataset=3, inplace=False)
#pd.DataFrame.to_csv(wvs_mice_comp_4, path_or_buf = '/data/wvs_impute_4.csv')
#wvs_mice_comp_5 = wvs_kernel.complete_data(dataset=4, inplace=False)
#pd.DataFrame.to_csv(wvs_mice_comp_5, path_or_buf = '/data/wvs_impute_5.csv')
#wvs_mice_comp_6 = wvs_kernel.complete_data(dataset=5, inplace=False)
#pd.DataFrame.to_csv(wvs_mice_comp_6, path_or_buf = '/data/wvs_impute_6.csv')
#wvs_mice_comp_7 = wvs_kernel.complete_data(dataset=6, inplace=False)
#pd.DataFrame.to_csv(wvs_mice_comp_7)
#wvs_mice_comp_8 = wvs_kernel.complete_data(dataset=7, inplace=False)
#wvs_mice_comp_9 = wvs_kernel.complete_data(dataset=8, inplace=False)
#wvs_mice_comp_10 = wvs_kernel.complete_data(dataset=9, inplace=False)
