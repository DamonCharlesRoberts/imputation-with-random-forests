# Title: Data Cleaning for RF MI

# Notes: 
    #* Description: Import and clean world values survey
    #* Updated: 2021-10-27
    #* Updated by: dcr
# Setup
    #* Load modules
import wget
wget.download('https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py')
import numpy as np #Numpy for arrays
import pandas as pd #Pandas for DataFrames
import torch #For produce_NA function
import matplotlib.pyplot as plt
import seaborn as sns #For heatmap
from utils import *

#import random as rand # Dropping random values in array
    #* Load world values data
wvs = pd.read_csv('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_original.csv')
    #* Define Ampute Function called produce_NA
def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}

# Clean 
    #* Set Seed
np.random.seed(601)
    #* Complete obs
wvs = wvs.drop(columns = ['version', 'doi', 'B_COUNTRY_ALPHA', 'C_COW_ALPHA', 'LNGE_ISO', 'CPARTY' ,'Partyname', 'Partyabb', 'CPARTYABB'])
wvs.dropna(axis = 0, how = 'any', inplace = True)
pd.DataFrame.to_csv(wvs, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_complete_obs_only.csv', index = True)
wvs = wvs.to_numpy()
    #* MCAR
wvs_sparse_mcar = produce_NA(wvs, p_miss = 0.4, mecha = "MCAR")
wvs_mcar = wvs_sparse_mcar['X_incomp']
np.savetxt('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_sparse_mcar.csv', wvs_mcar, delimiter =',')
    #* MAR
wvs_sparse_mar = produce_NA(wvs, p_miss = 0.4, mecha = "MAR", p_obs = 0.5)
wvs_mar = wvs_sparse_mar['X_incomp']
np.savetxt('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_sparse_mar.csv', wvs_mar, delimiter =',')
    #* MNAR
wvs_sparse_mnar = produce_NA(wvs, p_miss = 0.4, mecha = "MNAR", opt = 'logistic', p_obs = 0.5)
wvs_mnar = wvs_sparse_mnar['X_incomp']
np.savetxt('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_sparse_mnar.csv', wvs_mnar, delimiter = ',')