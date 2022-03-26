# Title: DCR Random Forest Imputation Primary Analysis

# Notes:
	#* Description: Performs primary analyses examining the performance of RF implementation of MICE to other missingness approaches
	#* Updated: 2022-03-26
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
anes = pd.read_csv('data/anes_2020_clean.csv', low_memory = False).drop(columns = ['Unnamed: 0'])
anes_imputed = pd.read_csv('data/anes_rf_imputed_3.csv').drop(columns = ['Unnamed: 0'])
anes_amelia = pd.read_csv('data/amelia_anes_partial.csv').drop(columns = ['Unnamed: 0'])

	#* RMSE
ANES_rmse = rmse(anes, anes_imputed)
ANES_rmse_2 = rmse(anes, anes_amelia)
headers = ['Data Set', 'RMSE', 'RMSE']
rmse = [('ANES - MICE', ANES_rmse), ('ANES - AMELIA II', ANES_rmse_2)]
print(tabulate(rmse, headers, tablefmt = 'latex', floatfmt = '.3f'))

	#* Regressions
		#** Recode presidential vote choice to dummy for democratic vote
anes['V202105x'] = anes['V202105x'].apply(lambda x: 1 if x == 10 else 0)
anes_imputed['V202105x'] = anes_imputed['V202105x'].apply(lambda x: 1 if x == 10 else 0)
anes_amelia['V202105x'] = anes_amelia['V202105x'].apply(lambda x: 1 if x == 10 else 0)
		#** Recode partisanship variable to be 1 - democrat, 2 - independent, 3 - republican
anes['V202065x'] = anes['V202065x'].apply(lambda x: 1 if x == 1 else(2 if x > 3 else 3))
anes_imputed['V202065x'] = anes_imputed['V202065x'].apply(lambda x: 1 if x == 1 else(2 if x > 3 else 3))
anes_amelia['V202065x'] = anes_amelia['V202065x'].apply(lambda x: 1 if x == 1 else(2 if x > 3 else 3))
		#** Recode race/ethnicity variable to white dummy
anes['V202065x'] = anes['V202065x'].apply(lambda x: 1 if x == 1 else(2 if x > 3 else 3))
anes_imputed['V201549x'] = anes_imputed['V201549x'].apply(lambda x: 1 if x == 1 else 0)
anes_amelia['V201549x'] = anes_amelia['V201549x'].apply(lambda x: 1 if x == 1 else 0)
		#** Recode sex variable to female dummy
anes['V201600'] = anes['V201600'].apply(lambda x: 1 if x == 2 else 0)
anes_imputed['V201600'] = anes_imputed['V201600'].apply(lambda x: 1 if x == 2 else 0)
anes_amelia['V201600'] = anes_amelia['V201600'].apply(lambda x: 1 if x == 2 else 0)

		#** Define regression function
def reg(df):
		df = df.rename(columns = {'V202105x': 'Democratic Vote Choice', 'V202065x':'Partisanship', 'V201511x': 'Education', 'V201507x':'Age', 'V201533x':'Occupational Status', 'V201549x':'White', 'V201600': 'Female'})
		x = df[['Partisanship', 'Education', 'Age', 'Occupational Status', 'White', 'Female']]
		x = sm.add_constant(x)
		y = df['Democratic Vote Choice']
		model = sm.OLS(y, x, missing = 'drop').fit()
		return(model)

		#** Run regression on the three datasets
anes_model = reg(anes)
anes_mice_model = reg(anes_imputed)
anes_amelia_model = reg(anes_amelia)

		#** Make tables displaying estimates from the three regressions
table1 = Stargazer([anes_mice_model, anes_amelia_model])
table2 = Stargazer([anes_model])
table1.show_confidence_intervals(True)
table1.custom_columns(['MICE Model - Vote Choice', 'AMELIA II Model - Vote Choice'],[1,1])
table1.show_model_numbers(False)
table1.add_custom_notes(['Data Source: 2020 American National Election Study', 'OLS Coefficients.', '95\\% Confidence Intervals in parentheses.'])
table2.show_confidence_intervals(True)
table2.custom_columns('LWD Model - Vote Choice')
table2.show_model_numbers(False)
table2.add_custom_notes(['Data Source: 2020 American National Election Study', 'OLS Coefficients.', '95\\% Confidence Intervals in parentheses.'])
print(table1.render_latex())
print(table2.render_latex())

		#** Make a coefficient plot of the three regressions
coefplot(anes_model)
plt.savefig('figures/anes_model.jpeg')
coefplot(anes_mice_model)
plt.savefig('figures/anes_mice_model.jpeg')
coefplot(anes_amelia_model)
plt.savefig('figures/anes_amelia_model.jpeg')

	#* Two independent sample t-test of coefficients
		#** Create dataframe of all the coefficients
anes_coef = anes_model.params
anes_coef = pd.DataFrame(anes_coef)
anes_coef['coefficientanes'] = anes_coef.iloc[:,0]
anes_coef = anes_coef.drop(anes_coef.columns[0], axis = 1)
anes_coef.index.name = 'index'
anes_coef = anes_coef.transpose()

anes_imputed_coef = anes_mice_model.params
anes_imputed_coef = pd.DataFrame(anes_imputed_coef)
anes_imputed_coef['coefficientMICE'] = anes_imputed_coef.iloc[:,0]
anes_imputed_coef = anes_imputed_coef.drop(anes_imputed_coef.columns[0], axis = 1)
anes_imputed_coef.index.name = 'index'
anes_imputed_coef = anes_imputed_coef.transpose()

anes_amelia_coef = anes_amelia_model.params
anes_amelia_coef = pd.DataFrame(anes_amelia_coef)
anes_amelia_coef['coefficientAMELIA'] = anes_amelia_coef.iloc[:,0]
anes_amelia_coef = anes_amelia_coef.drop(anes_amelia_coef.columns[0], axis = 1)
anes_amelia_coef.index.name = 'index'
anes_amelia_coef = anes_amelia_coef.transpose()

anes_coef_sample = pd.concat([anes_coef, anes_imputed_coef, anes_amelia_coef])
anes_coef_sample = anes_coef_sample.transpose()

		#** Comparison between ANES LWD and MICE
anes_coef_sample_x = anes_coef_sample
anes_coef_sample_x['difference'] = anes_coef_sample_x['coefficientanes'] - anes_coef_sample_x['coefficientMICE']
anes_coef_sample_x['beta1se'] = anes_model.bse
anes_coef_sample_x['beta2se'] = anes_mice_model.bse
anes_coef_sample_x['denom'] = np.sqrt((anes_coef_sample_x['beta1se'] **2) + (anes_coef_sample_x['beta2se'] **2))
anes_coef_sample_x['tstat'] = anes_coef_sample_x['difference']/anes_coef_sample_x['denom']
			#*** Make table of test
header = ['Variable', 'T-Statistic']
tstat = [('Constant', anes_coef_sample_x.iloc[0]['tstat']), 
('Partisanship', anes_coef_sample_x.iloc[1]['tstat']), 
('Education', anes_coef_sample_x.iloc[2]['tstat']),
('Age', anes_coef_sample_x.iloc[3]['tstat']),
('Occupational Status', anes_coef_sample_x.iloc[4]['tstat']),
('White', anes_coef_sample_x.iloc[5]['tstat']),
('Age', anes_coef_sample_x.iloc[6]['tstat'])]
print(tabulate(tstat, header, tablefmt = 'latex', floatfmt = '.3f'))

		#** Comparison between ANES LWD and AMELIA II
anes_coef_sample_x = anes_coef_sample
anes_coef_sample_x['difference'] = anes_coef_sample_x['coefficientanes'] - anes_coef_sample_x['coefficientAMELIA']
anes_coef_sample_x['beta1se'] = anes_model.bse
anes_coef_sample_x['beta2se'] = anes_mice_model.bse
anes_coef_sample_x['denom'] = np.sqrt((anes_coef_sample_x['beta1se'] **2) + (anes_coef_sample_x['beta2se'] **2))
anes_coef_sample_x['tstat'] = anes_coef_sample_x['difference']/anes_coef_sample_x['denom']
			#** Make table of comparison
header = ['Variable', 'T-Statistic']
tstat = [('Constant', anes_coef_sample_x.iloc[0]['tstat']), 
('Partisanship', anes_coef_sample_x.iloc[1]['tstat']), 
('Education', anes_coef_sample_x.iloc[2]['tstat']),
('Age', anes_coef_sample_x.iloc[3]['tstat']),
('Occupational Status', anes_coef_sample_x.iloc[4]['tstat']),
('White', anes_coef_sample_x.iloc[5]['tstat']),
('Age', anes_coef_sample_x.iloc[6]['tstat'])]
print(tabulate(tstat, header, tablefmt = 'latex', floatfmt = '.3f'))

		#** Comparison between MICE and AMELIA II
anes_coef_sample_x = anes_coef_sample
anes_coef_sample_x['difference'] = anes_coef_sample_x['coefficientMICE'] - anes_coef_sample_x['coefficientAMELIA']
anes_coef_sample_x['beta1se'] = anes_model.bse
anes_coef_sample_x['beta2se'] = anes_mice_model.bse
anes_coef_sample_x['denom'] = np.sqrt((anes_coef_sample_x['beta1se'] **2) + (anes_coef_sample_x['beta2se'] **2))
anes_coef_sample_x['tstat'] = anes_coef_sample_x['difference']/anes_coef_sample_x['denom']
			#*** Make table of comparison
header = ['Variable', 'T-Statistic']
tstat = [('Constant', anes_coef_sample_x.iloc[0]['tstat']), 
('Partisanship', anes_coef_sample_x.iloc[1]['tstat']), 
('Education', anes_coef_sample_x.iloc[2]['tstat']),
('Age', anes_coef_sample_x.iloc[3]['tstat']),
('Occupational Status', anes_coef_sample_x.iloc[4]['tstat']),
('White', anes_coef_sample_x.iloc[5]['tstat']),
('Age', anes_coef_sample_x.iloc[6]['tstat'])]
print(tabulate(tstat, header, tablefmt = 'latex', floatfmt = '.3f'))