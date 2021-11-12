# Title: Data Cleaning and Simulation for RF MI

# Notes: 
	#* Description: Generate simulated datasets and import and clean world values survey
	#* Updated: 2021-10-27
	#* Updated by: dcr
# Setup
	#* Load modules
import numpy as np #Numpy for arrays
import pandas as pd #Pandas for DataFrames

    #* Source produce_na_funciton.py script for produce_NA fxn
exec(open('code/produce_na_function.py').read())



# Generate simulated data
	#* Set Seed
np.random.seed(601)
	#* Three variables
mean1 = np.random.random(3) # Randomly generate the means for 2 variables
cov1 = np.random.random((3,3)) # Randomly generate the covariances for 3 variables
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
sim1df = pd.DataFrame(sim1, columns = ['Column_A', 'Column_B', 'Column_C'])
pd.DataFrame.to_csv(sim1df, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary.csv', index = True)
sim1_mar = pd.DataFrame(sim1_mar, columns = ['Column_A', 'Column_B', 'Column_C'])
pd.DataFrame.to_csv(sim1_mar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary_sparse_mar.csv', index = True)
sim1_mcar = pd.DataFrame(sim1_mcar, columns = ['Column_A', 'Column_B', 'Column_C'])
pd.DataFrame.to_csv(sim1_mcar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary_sparse_mcar.csv', index = True)
sim1_mnar = pd.DataFrame(sim1_mnar, columns = ['Column_A', 'Column_B', 'Column_C'])
pd.DataFrame.to_csv(sim1_mnar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary_sparse_mnar.csv', index = True)
sim10 = pd.DataFrame(sim10, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim.csv', index = True)
sim10_mcar = pd.DataFrame(sim10_mcar, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10_mcar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim_sparse_mcar.csv', index = True)
sim10_mar = pd.DataFrame(sim10_mar, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10_mar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim_sparse_mar.csv', index = True)
sim10_mnar = pd.DataFrame(sim10_mnar, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10_mnar, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim_sparse_mnar.csv', index = True)