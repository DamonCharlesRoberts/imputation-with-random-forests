# Title: DCR Random Forest Imputation Primary Analysis

# Notes:
	#* Description: Performs primary analyses examining the performance of RF implementation of MICE to other missingness approaches
	#* Updated: 2021-11-12
	#* Updated by: dcr
# Setup:
	#* Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statistics import median
from stargazer.stargazer import Stargazer
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
	#* Working Directory
os.chdir('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/') # MAC
#os.chdir('/home/damoncroberts/Dropbox/current_projects/dcr_rf_imputation') #LINUX
	#* Source rmse function file
exec(open('code/rmse_function.py').read())
	#* Source coefplot function file
exec(open('code/coef_plot_fxn.py').read())
	#* Load Data
wvs = pd.read_csv('data/wvs_original.csv', low_memory = False)
wvs = wvs[wvs['A_YEAR'] == 2018].drop(columns = ['version', 'doi', 'A_WAVE', 'A_STUDY', 'B_COUNTRY', 'B_COUNTRY_ALPHA', 'C_COW_NUM', 'C_COW_ALPHA', 'A_YEAR', 'D_INTERVIEW', 'J_INTDATE', 'FW_END', 'FW_START', 'K_TIME_START', 'K_TIME_END', 'K_DURATION', 'Q_MODE', 'N_REGION_ISO', 'N_REGION_WVS', 'N_TOWN', 'O1_LONGITUDE', 'O2_LATITUDE', 'S_INTLANGUAGE', 'LNGE_ISO', 'E_RESPINT', 'F_INTPRIVACY', 'E1_LITERACY', 'W_WEIGHT', 'S018', 'PWGHT', 'S025', 'Partyname', 'Partyabb', 'CPARTY', 'CPARTYABB']).rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).reset_index().dropna()
wvs_imputed_3 = pd.read_csv('data/wvs_impute_3.csv').drop(columns = ['Unnamed: 0'])
wvs_imputed_0 = pd.read_csv('data/wvs_impute_0.csv').drop(columns = ['Unnamed: 0'])
wvs_imputed_1 = pd.read_csv('data/wvs_impute_1.csv').drop(columns = ['Unnamed: 0'])
wvs_imputed_2 = pd.read_csv('data/wvs_impute_2.csv').drop(columns = ['Unnamed: 0'])

	#* RMSE
wvs_rmse = rmse(wvs, wvs_imputed_3)
headers = ['Data Set', 'RMSE']
rmse = [('WVS - Average', wvs_rmse)]
print(tabulate(rmse, headers, tablefmt = 'latex', floatfmt = '.3f'))

	#* Regressions
def reg(df):
		df['membership'] = df.iloc[:, 118:129].sum(axis=1)
		df = df.rename(columns = {'Q289': 'Income', 'Q112': 'Clientelism', 'Q118': 'Corrupt', 'Q234': 'Fair Election', 'membership': 'Membership'})
		x = df[['Income', 'Clientelism', 'Membership', 'Corrupt', 'Fair Election']]
		x = sm.add_constant(x)
		y = df['Q222']
		model = sm.OLS(y, x).fit()
		return(model)

wvs_imputed_0_model = reg(wvs_imputed_0)
wvs_imputed_1_model = reg(wvs_imputed_1)
wvs_imputed_2_model = reg(wvs_imputed_2)
wvs_imputed_3_model = reg(wvs_imputed_3)
wvs_model = reg(wvs)

table1 = Stargazer([wvs_imputed_0_model, wvs_imputed_1_model])
table2 = Stargazer([wvs_imputed_2_model, wvs_imputed_3_model])
table3 = Stargazer([wvs_model])
table1.show_confidence_intervals(True)
table1.custom_columns(['Imputed Model 1', 'Imputed Model 2'],[1,1])
table1.show_model_numbers(False)
table1.add_custom_notes(['Data Source: World Values Survey 2018 Panel', 'OLS Coefficients.', '95\\% Confidence Intervals in parentheses.'])
table2.show_confidence_intervals(True)
table2.custom_columns(['Imputed Model 3', 'Imputed Model 4'], [1,1])
table2.show_model_numbers(False)
table2.add_custom_notes(['Data Source: World Values Survey 2018 Panel', 'OLS Coefficients.', '95\\% Confidence Intervals in parentheses.'])
table3.show_confidence_intervals(True)
table3.custom_columns('LWD Model')
table3.show_model_numbers(False)
table3.add_custom_notes(['Data Source: World Values Survey 2018 Panel', 'OLS Coefficients.', '95\\% Confidence Intervals in parentheses.'])
print(table1.render_latex())
print(table2.render_latex())
print(table3.render_latex())


coefplot(wvs_model)
plt.savefig('figures/wvs_model.jpeg')
coefplot(wvs_imputed_0_model)
plt.savefig('figures/wvs_imputed_0_model.jpeg')
coefplot(wvs_imputed_1_model)
plt.savefig('figures/wvs_imputed_1_model.jpeg')
coefplot(wvs_imputed_2_model)
plt.savefig('figures/wvs_imputed_2_model.jpeg')
coefplot(wvs_imputed_3_model)
plt.savefig('figures/wvs_imputed_3_model.jpeg')
		#* Two independent sample t-test of coefficients

wvs_coef = wvs_model.params
wvs_coef = pd.DataFrame(wvs_coef)
wvs_coef['coefficientwvs'] = wvs_coef.iloc[:,0]
wvs_coef = wvs_coef.drop(wvs_coef.columns[0], axis = 1)
wvs_coef.index.name = 'index'
wvs_coef = wvs_coef.transpose()
wvs_imputed_0_coef = wvs_imputed_0_model.params
wvs_imputed_0_coef = pd.DataFrame(wvs_imputed_0_coef)
wvs_imputed_0_coef['coefficient1'] = wvs_imputed_0_coef.iloc[:,0]
wvs_imputed_0_coef = wvs_imputed_0_coef.drop(wvs_imputed_0_coef.columns[0], axis = 1)
wvs_imputed_0_coef.index.name = 'index'
wvs_imputed_0_coef = wvs_imputed_0_coef.transpose()
wvs_imputed_1_coef = wvs_imputed_1_model.params
wvs_imputed_1_coef = pd.DataFrame(wvs_imputed_1_coef)
wvs_imputed_1_coef['coefficient2'] = wvs_imputed_1_coef.iloc[:,0]
wvs_imputed_1_coef = wvs_imputed_1_coef.drop(wvs_imputed_1_coef.columns[0], axis = 1)
wvs_imputed_1_coef.index.name = 'index'
wvs_imputed_1_coef = wvs_imputed_1_coef.transpose()
wvs_imputed_2_coef = wvs_imputed_2_model.params
wvs_imputed_2_coef = pd.DataFrame(wvs_imputed_2_coef)
wvs_imputed_2_coef['coefficient3'] = wvs_imputed_2_coef.iloc[:,0]
wvs_imputed_2_coef = wvs_imputed_2_coef.drop(wvs_imputed_2_coef.columns[0], axis = 1)
wvs_imputed_2_coef.index.name = 'index'
wvs_imputed_2_coef = wvs_imputed_2_coef.transpose()
wvs_imputed_3_coef = wvs_imputed_3_model.params
wvs_imputed_3_coef = pd.DataFrame(wvs_imputed_3_coef)
wvs_imputed_3_coef['coefficient4'] = wvs_imputed_3_coef.iloc[:,0]
wvs_imputed_3_coef = wvs_imputed_3_coef.drop(wvs_imputed_3_coef.columns[0], axis = 1)
wvs_imputed_3_coef.index.name = 'index'
wvs_imputed_3_coef = wvs_imputed_3_coef.transpose()


wvs_coef_sample = pd.concat([wvs_coef, wvs_imputed_0_coef, wvs_imputed_1_coef, wvs_imputed_2_coef, wvs_imputed_3_coef])
wvs_coef_sample = wvs_coef_sample.transpose()

wvs_coef_sample_x = wvs_coef_sample
wvs_coef_sample_x['difference'] = wvs_coef_sample_x['coefficientwvs'] - wvs_coef_sample_x['coefficient4']
wvs_coef_sample_x['beta1se'] = wvs_model.bse
wvs_coef_sample_x['beta2se'] = wvs_imputed_3_model.bse
wvs_coef_sample_x['denom'] = np.sqrt((wvs_coef_sample_x['beta1se'] **2) + (wvs_coef_sample_x['beta2se'] **2))
wvs_coef_sample_x['tstat'] = wvs_coef_sample_x['difference']/wvs_coef_sample_x['denom']

header = ['Variable', 'T-Statistic']
tstat = [('Constant', wvs_coef_sample_x.iloc[0]['tstat']), 
('Income', wvs_coef_sample_x.iloc[1]['tstat']), 
('Clientelism', wvs_coef_sample_x.iloc[2]['tstat']),
('Membership', wvs_coef_sample_x.iloc[3]['tstat']),
('Public Officials Corrupt', wvs_coef_sample_x.iloc[4]['tstat']),
('Fair Election', wvs_coef_sample_x.iloc[5]['tstat'])]
print(tabulate(tstat, header, tablefmt = 'latex', floatfmt = '.3f'))