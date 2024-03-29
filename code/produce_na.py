# Title: Define Produce_NA function

# Notes:
	#* Description: Script to define Produce_NA function
	#* Updated: 2023-01-12
	#* Updated by: dcr
# Import modules
    #* env
import os
import torch #For produce_NA function
import wget
    #* download utils
URL = 'https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py'
FILE_NAME = '../code/utils.py'
if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)
wget.download(URL, out=FILE_NAME)

#wget.download('https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py')
from utils import *

# Define Ampute Function called produce_NA
def produce_na(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
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
           Indicates the missing-data mechanism to be used.
           "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str,
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated using a:
            - logistic regression ("logistic")
            - quantile censorship ("quantile")
            - logistic regression for generating a self-masked MNAR mechanism ("selfmasked")
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti":
                - proportion of variables with *no* missing values
                    - will be used for the logistic masking model.
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
        X = X.to_numpy()
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