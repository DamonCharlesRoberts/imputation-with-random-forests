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
import scipy as sp
from scipy.stats import chi2
	#* Working Directory
os.chdir('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/')
	#* Source rmse function file
exec(open('code/rmse_function.py').read())
	#* Load Data
wvs = pd.read_csv('data/wvs_original.csv', low_memory = False)
wvs = wvs[wvs['A_YEAR'] == 2018].drop(columns = ['version', 'doi', 'A_WAVE', 'A_STUDY', 'B_COUNTRY', 'B_COUNTRY_ALPHA', 'C_COW_NUM', 'C_COW_ALPHA', 'A_YEAR', 'D_INTERVIEW', 'J_INTDATE', 'FW_END', 'FW_START', 'K_TIME_START', 'K_TIME_END', 'K_DURATION', 'Q_MODE', 'N_REGION_ISO', 'N_REGION_WVS', 'N_TOWN', 'O1_LONGITUDE', 'O2_LATITUDE', 'S_INTLANGUAGE', 'LNGE_ISO', 'E_RESPINT', 'F_INTPRIVACY', 'E1_LITERACY', 'W_WEIGHT', 'S018', 'PWGHT', 'S025', 'Partyname', 'Partyabb', 'CPARTY', 'CPARTYABB']).rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).reset_index()
wvs_imputed_3 = pd.read_csv('data/wvs_impute_3.csv').drop(columns = ['Unnamed: 0'])
wvs_imputed_0 = pd.read_csv('data/wvs_impute_0.csv').drop(columns = ['Unnamed: 0'])
wvs_imputed_1 = pd.read_csv('data/wvs_impute_1.csv').drop(columns = ['Unnamed: 0'])
wvs_imputed_2 = pd.read_csv('data/wvs_impute_2.csv').drop(columns = ['Unnamed: 0'])

	#* RMSE
wvs_rmse = rmse(wvs, wvs_impute3)
headers = ['Data Set', 'RMSE']
rmse = [('WVS - Average', wvs_rmse)]
print(tabulate(rmse, headers, tablefmt = 'latex', floatfmt = '.3f'))

	#* Regressions
datasets = [wvs_imputed_0, wvs_imputed_1, wvs_imputed_2, wvs_imputed_3, wvs]
def reg(df):
		df['membership'] = df.iloc[:, 118:129].sum(axis=1)
		x = df[['Q289', 'Q112', 'membership', 'Q118', 'Q234']]
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
table1.render_latex()
table2.render_latex()
table3.render_latex()
		#* Malahanobis distance

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

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

wvs_coef_sample_x = wvs_coef_sample[['coefficientwvs', 'coefficient1', 'coefficient2', 'coefficient3', 'coefficient4']]
wvs_coef_sample_x['mahala'] = mahalanobis(x = wvs_coef_sample_x, data = wvs_coef_sample[['coefficientwvs', 'coefficient1', 'coefficient2', 'coefficient3', 'coefficient4']])
wvs_coef_sample_x['p_value'] = 1 - chi2.cdf(wvs_coef_sample_x['mahala'], 2)