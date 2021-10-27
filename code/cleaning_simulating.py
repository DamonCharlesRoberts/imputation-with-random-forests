# Title: Data Cleaning and Simulation for RF MI

# Notes: 
	#* Description: Generate simulated datasets and import and clean world values survey
	#* Updated: 2021-10-27
	#* Updated by: dcr
# Setup
	#* Load modules
import numpy as np
import pandas as pd
import random as rand
	#* Load world values data
wvs = pd.read_csv('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/wvs_original.csv')
# Generate simulated data
	#* Two variables
mean1 = np.random.random(2)
cov1 = np.random.random((2,2))
sim1 = np.random.multivariate_normal(mean1, cov1, 1000)
sim1_sparse_i = rand.randint(0, len(sim1))
sim1_sparse_set = sim1[sim1_sparse_i]
sim1_sparse = np.delete(sim1, np.arange(sim1_sparse_i, sim1_sparse_i+len(sim1_sparse_set),1)).reshape([-1, len(sim1_sparse_set)])
	#* 10 Variables
mean10 = np.random.random(10)
cov10 = np.random.random((10, 10))
#mean10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
#cov10 = [[1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0]]
sim10 = np.random.multivariate_normal(mean10, cov10, 1000)
sim10_sparse_i = rand.randint(0, len(sim10))
sim10_sparse_set = sim10[sim10_sparse_i]
sim10_sparse = np.delete(sim10, np.arange(sim10_sparse_i, sim10_sparse_i+len(sim10_sparse_set),1)).reshape([-1, len(sim10_sparse_set)])


# Save Simulated and Cleaned data
	#* Simulated
sim1df = pd.DataFrame(sim1, columns = ['Column_A', 'Column_B'])
pd.DataFrame.to_csv(sim1df, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary.csv', index = True)
sim1_sparse = pd.DataFrame(sim1_sparse, columns = ['Column_A', 'Column_B'])
pd.DataFrame.to_csv(sim1_sparse, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_binary_sparse.csv', index = True)
sim10 = pd.DataFrame(sim10, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim.csv', index = True)
sim10_sparse = pd.DataFrame(sim10, columns = ['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_G', 'Column_H', 'Column_I', 'Column_J'])
pd.DataFrame.to_csv(sim10_sparse, path_or_buf='/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ndim_sparse.csv', index = True)
