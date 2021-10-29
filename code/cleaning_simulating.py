# Title: Data Cleaning and Simulation for RF MI

# Notes: 
	#* Description: Generate simulated datasets and import and clean world values survey
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

# Generate simulated data
	#* Set Seed
np.random.seed(601)
	#* Two variables
mean1 = np.random.random(2) # Randomly generate the means for 2 variables
cov1 = np.random.random((2,2)) # Randomly generate the covariances for 2 variables
sim1 = np.random.multivariate_normal(mean1, cov1, 1000) #Use mean and covariates to generate a simulated dataset following a multivariate normal distribution with an N = 1000
#sim1 = pd.DataFrame(sim1, columns = ['Column_A', ['Column_B']]) # Convert this simulated dataset into a dataframe and give it column names
	#* MCAR
sim1_sparse_mcar = produce_NA(sim1, p_miss=0.4, mecha="MCAR")
 # randomly drop 40% of rows in the dataframe
sim1_mcar = sim1_sparse_mcar['X_incomp']
sim1_mcar_r = sim1_sparse_mcar['mask']
	#* MAR
sim1_sparse_mar = produce_NA(sim1, p_miss=0.4, mecha="MAR", p_obs =0.5)
sim1_mar = sim1_sparse_mar['X_incomp']
	#* MNAR
sim1_sparse_mnar = produce_NA(sim1, p_miss=0.4, mecha="MNAR", opt="logistic", p_obs=0.5)
sim1_mnar = sim1_sparse_mnar['X_incomp']
#sim1_sparse_i = rand.randint(0, len(sim1))
#sim1_sparse_set = sim1[sim1_sparse_i]
#sim1_sparse = np.delete(sim1, np.arange(sim1_sparse_i, sim1_sparse_i+len(sim1_sparse_set),1)).reshape([-1, len(sim1_sparse_set)])
	#* 10 Variables
mean10 = np.random.random(10) # Randomly generate the means for 10 variables
cov10 = np.random.random((10, 10)) # Randomly generate the covariances for 10 variables
#mean10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
#cov10 = [[1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0]]
sim10 = np.random.multivariate_normal(mean10, cov10, 1000) #Use mean and covariates to generate a simulated dataset following a multivariate normal distribution with an N = 1000
#sim10 = pd.DataFrame(sim10, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J']) # Convert this simulated dataset into a dataframe and give it column names
#sim10_sparse = sim10.sample(frac = 0.2, state = 0601) # randomly drop 20% of rows in the dataframe
#sim10_sparse_i = rand.randint(0, len(sim10))
#sim10_sparse_set = sim10[sim10_sparse_i]
#sim10_sparse = np.delete(sim10, np.arange(sim10_sparse_i, sim10_sparse_i+len(sim10_sparse_set),1)).reshape([-1, len(sim10_sparse_set)])
	#* MCAR
sim10_sparse_mcar = produce_NA(sim10, p_miss = 0.4, mecha = "MCAR")
sim10_mcar = sim10_sparse_mcar['X_incomp']
	#* MAR
sim10_sparse_mar = produce_NA(sim10, p_miss = 0.4, mecha = "MAR", p_obs = 0.5)
sim10_mar = sim10_sparse_mar['X_incomp']
	#* MNAR
sim10_sparse_mnar = produce_NA(sim10, p_miss = 0.4, mecha = "MNAR", opt = "logistic", p_obs=0.5)
sim10_mnar = sim10_sparse_mnar['X_incomp']
# Save Simulated and Cleaned data
	#* Simulated
sim1df = pd.DataFrame(sim1, columns = ['Column_A', 'Column_B'])
pd.DataFrame.to_csv(sim1df, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary.csv', index = True)
sim1_mar = pd.DataFrame(sim1_mar, columns = ['Column_A', 'Column_B'])
pd.DataFrame.to_csv(sim1_mar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary_sparse_mar.csv', index = True)
sim1_mcar = pd.DataFrame(sim1_mcar, columns = ['Column_A', 'Column_B'])
pd.DataFrame.to_csv(sim1_mcar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary_sparse_mcar.csv', index = True)
sim1_mnar = pd.DataFrame(sim1_mnar, columns = ['Column_A', 'Column_B'])
pd.DataFrame.to_csv(sim1_mnar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary_sparse_mnar.csv', index = True)
sim10 = pd.DataFrame(sim10, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim.csv', index = True)
sim10_mcar = pd.DataFrame(sim10_mcar, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10_mcar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim_sparse_mcar.csv', index = True)
sim10_mar = pd.DataFrame(sim10_mar, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10_mar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim_sparse_mar.csv', index = True)
sim10_mnar = pd.DataFrame(sim10_mnar, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10_mnar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim_sparse_mnar.csv', index = True)