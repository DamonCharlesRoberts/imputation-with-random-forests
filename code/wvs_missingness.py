# Title: Data Cleaning for RF MI

# Notes: 
    #* Description: Import and clean world values survey
    #* Updated: 2021-10-27
    #* Updated by: dcr
# Setup
    #* Load modules
import numpy as np #Numpy for arrays
import pandas as pd #Pandas for DataFrames


    #* Source produce_na_funciton.py script for produce_NA fxn
exec(open('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/code/produce_na_function.py').read())

#import random as rand # Dropping random values in array
    #* Load world values data
wvs = pd.read_csv('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_original.csv')

# Clean 
    #* Set Seed
np.random.seed(601)
    #* Complete obs
wvs = wvs.drop(columns = ['version', 'doi', 'B_COUNTRY_ALPHA', 'C_COW_ALPHA', 'LNGE_ISO', 'CPARTY' ,'Partyname', 'Partyabb', 'CPARTYABB'])
wvs.dropna(axis = 0, how = 'any', inplace = True)
pd.DataFrame.to_csv(wvs, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_complete_obs_only.csv', index = True)
wvs_colnames= wvs.keys()
wvs_numpy = wvs.to_numpy()


class wvs_missingness:
        def __init__(wvs_i, types):
            if types == 'MCAR':
                wvs_i = produce_NA(wvs_numpy, p_miss = 0.4, mecha = types)
            elif types == 'MAR':
                wvs_i = produce_NA(wvs_numpy, p_miss = 0.4, mecha = types, p_obs = 0.5)
            elif types == 'MNAR':
                wvs_i = produce_NA(wvs_numpy, p_miss = 0.4, mecha = types, p_obs = 0.5, opt = 'logistic')
            else:
                print('Invalid argument')
            wvs_i = wvs_i['X_incomp']
            wvs_i = pd.DataFrame(wvs_i.numpy(), columns = wvs_colnames)
            pd.DataFrame.to_csv(wvs_i, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_sparse_' + types.lower() + '.csv', index = True) 
wvs_missingness(types = 'MCAR')
wvs_missingness(types = 'MAR')
wvs_missingness(types = 'MNAR')
    #* MCAR - All Variables
#wvs_mcar = produce_NA(wvs_numpy, p_miss = 0.4, mecha = "MCAR")
#wvs_mcar = wvs_mcar['X_incomp']
#wvs_mcar = pd.DataFrame(wvs_mcar, columns = wvs_colnames)
#pd.DataFrame.to_csv(wvs_mcar, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_sparse_mcar.csv', index = True)
    #* MAR - All Variables 
#wvs_mar = produce_NA(wvs_numpy, p_miss = 0.4, mecha = "MAR", p_obs = 0.5)
#wvs_mar = wvs_mar['X_incomp']
#wvs_mar = pd.DataFrame(wvs_mar, columns = wvs_colnmaes)
#pd.DataFrame.to_csv(wvs_mar, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_sparse_mar.csv', index = True)
    #* MNAR - All Variables
#wvs_mnar = produce_NA(wvs_numpy, p_miss = 0.4, mecha = "MNAR", opt = 'logistic', p_obs = 0.5)
#wvs_mnar = wvs_mnar['X_incomp']
#wvs_mnar = pd.DataFrame(wvs_mnar, columns = wvs_colnames)
#pd.DataFrame.to_csv(wvs_mcar, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_sparse_mnar.csv', index = True)

    #* Make a k = 10 dataframe
wvs_low_k = wvs[['Q222', 'Q289', 'Q112', 'Q118', 'Q234', 'Q94', 'Q95', 'Q96', 'Q97', 'Q98', 'Q99', 'Q100', 'Q101', 'Q102', 'Q103', 'Q104', 'Q105']]
pd.DataFrame.to_csv(wvs_low_k, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_complete_obs_low_k.csv', index = True)
wvs_low_k_colnames = wvs_low_k.keys()
wvs_low_k_numpy = wvs_low_k.to_numpy()

class wvs_low_k_missingness:
        def __init__(wvs_low_k_i, types):
            if types == 'MCAR':
                wvs_low_k_i = produce_NA(wvs_low_k_numpy, p_miss = 0.4, mecha = types)
            elif types == 'MAR':
                wvs_low_k_i = produce_NA(wvs_low_k_numpy, p_miss = 0.4, mecha = types, p_obs = 0.5)
            elif types == 'MNAR':
                wvs_low_k_i = produce_NA(wvs_low_k_numpy, p_miss = 0.4, mecha = types, p_obs = 0.5, opt = 'logistic')
            else:
                print('Invalid argument')
            wvs_low_k_i = wvs_low_k_i['X_incomp']
            wvs_low_k_i = pd.DataFrame(wvs_low_k_i.numpy(), columns = wvs_low_k_colnames)
            pd.DataFrame.to_csv(wvs_low_k_i, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_low_k_' + types.lower() + '.csv', index = True) 
wvs_low_k_missingness(types = 'MCAR')
wvs_low_k_missingness(types = 'MAR')
wvs_low_k_missingness(types = 'MNAR')
    #* MCAR - Low K Variables
#wvs_low_k_mcar = produce_NA(wvs_low_k_numpy, p_miss = 0.4, mecha = 'MCAR')
#wvs_low_k_mcar = wvs_low_k_mcar['X_incomp']
#wvs_low_k_mcar = pd.DataFrame(wvs_low_k_mcar, columns = wvs_low_k_colnames)
#pd.DataFrame.to_csv(wvs_low_k_mcar, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_low_k_mcar.csv', index = True)
    #* MAR - Low K Variables
#wvs_low_k_mar = produce_NA(wvs_low_k_numpy, p_miss = 0.4, mecha = 'MAR')
#wvs_low_k_mar = wvs_low_k_mar['X_incomp']
#wvs_low_k_mar = pd.DataFrame(wvs_low_k_mar, columns = wvs_low_k_colnames)
#pd.DataFrame.to_csv(wvs_low_k_mar, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_low_k_mar.csv', index = True)
    #* MNAR - Low K Variables
#wvs_low_k_mnar = produce_NA(wvs_low_k_numpy, p_miss = 0.4, mecha = 'MNAR')
#wvs_low_k_mnar = wvs_low_k_mnar['X_incomp']
#wvs_low_k_mnar = pd.DataFrame(wvs_low_k_mnar, columns = wvs_low_k_colnames)
#pd.DataFrame.to_csv(wvs_low_k_mnar, path_or_buf = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputationdata/wvs_low_k_mnar.csv', index = True)