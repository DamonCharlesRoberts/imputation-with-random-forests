# Title: Ampute the simulated dataset with MAR pattern

# Notes:
    #* Description: Python script using produce_na function to introduce MAR missingness into simulated data
    #* Updated: 2022 - 08-12
    #* Updated by: dcr

# Setup
    #* Load modules
import numpy as np #Numpy for arrays
import pandas as pd #Pandas for DataFrames
    #* Set the seed
np.random.seed(90210)
    #* Source produce_na_function.py script for produce_na fxn
exec(open('/home/damoncroberts/Dropbox/current_projects/dcr_rf_imputation/drafts/produce_na.py').read())
    #* Generate data
A = np.random.binomial(n = 500, p = 0.5)
B = np.random.uniform(size = 500)
X = np.random.normal(size = 500)
Z = np.random.binomial(n = 500, p = 0.7)
#mean = np.random.random(9) # Randomly generate the means for 10 variables
#cov = np.random.random((9, 9)) # Randomly generate the covariances for 10 variables
#sim = np.random.multivariate_normal(mean, cov, 500) #Use mean and covariates to generate a simulated dataset following a multivariate normal distribution with an N = 1000
#sim = pd.DataFrame(sim, columns = ['X', 'Z', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])
sim = pd.DataFrame({'A': A, 'B':B, 'X':X, 'Z':Z}, columns = ['A', 'B', 'X', 'Z'])
sim['Y_continuous'] = sim['X'] + sim['Z'] + np.random.normal(0, 0.1, 500)
sim['Y_ordinal'] = pd.cut(sim['Y_continuous'], bins = 5, labels = False)
sim = sim.drop(columns=['Y_continuous'])
sim.to_csv('data/temp_complete.csv')
# Ampute the data
b = produce_na(sim, p_miss = 0.4, mecha = 'MAR', p_obs = 0.5)
b = b['X_incomp']
b = pd.DataFrame(b.numpy(), columns = ['Y_ordinal', 'X', 'Z', 'A', 'B'])
b.to_csv('data/temp_amputed.csv')

a = produce_na(sim, p_miss = 0.1, mecha = 'MAR', p_obs = 0.5)
a = a['X_incomp']
a = pd.DataFrame(a.numpy(), columns = ['Y_ordinal', 'X', 'Z', 'A', 'B'])
a.to_csv('data/temp_amputed_0_1.csv')

c = produce_na(sim, p_miss = 0.9, mecha = 'MAR', p_obs = 0.5)
c = c['X_incomp']
c = pd.DataFrame(c.numpy(), columns = ['Y_ordinal', 'X', 'Z', 'A', 'B'])
c.to_csv('data/temp_amputed_0_9.csv')