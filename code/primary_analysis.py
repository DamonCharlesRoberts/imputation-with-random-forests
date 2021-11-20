# Title: DCR Random Forest Imputation Primary Analysis

# Notes:
	#* Description: Performs primary analyses examining the performance of RF implementation of MICE to other missingness approaches
	#* Updated: 2021-11-12
	#* Updated by: dcr
# Setup:
	#* Libraries
import miceforest as mf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from tabulate import tabulate
import os
	#* Plot Style 
sns.set(style = 'whitegrid')
	#* Working directory
#os.chdir('/home/damoncroberts/Dropbox/current_projects/dcr_rf_imputation/') # Linux
os.chdir('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/') # Mac
# Simulated Data 
	#* Import complete data
sim_3_comp = pd.read_csv(filepath_or_buffer = 'data/sim_three_complete.csv')
sim_3_comp = sim_3_comp.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
sim_10_comp = pd.read_csv(filepath_or_buffer = 'data/sim_ten_complete.csv')
sim_10_comp = sim_10_comp.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
wvs_comp = pd.read_csv(filepath_or_buffer = 'data/wvs_complete_obs_only.csv')
	#* Import MAR data
sim_3_mar = pd.read_csv(filepath_or_buffer = 'data/sim_three_MAR.csv')
sim_10_mar = pd.read_csv(filepath_or_buffer = 'data/sim_ten_MAR.csv')
wvs_mar = pd.read_csv(filepath_or_buffer = 'data/wvs_sparse_MAR.csv')
wvs_low_k_mar = pd.read_csv(filepath_or_buffer = 'data/wvs_low_k_MAR.csv')
	#* Import MCAR data
#sim_3_mcar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_MCAR.csv')
#sim_10_mcar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_MCAR.csv')
	#* Import MNAR data
#sim_3_mnar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_MNAR.csv')
#sim_10_mnar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_MNAR.csv')

	#* 3 variable

sim_3_mar = sim_3_mar.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

sim_3_kernel = mf.ImputationKernel(
                                   sim_3_mar,
                                   datasets=10,
                                   save_all_iterations=True,
                                   random_state=601)
sim_3_kernel.mice(10)

sim_3_mice_comp_1 = sim_3_kernel.complete_data(dataset=0, inplace=False)
sim_3_mice_comp_2 = sim_3_kernel.complete_data(dataset=1, inplace=False)
sim_3_mice_comp_3 = sim_3_kernel.complete_data(dataset=2, inplace=False)
sim_3_mice_comp_4 = sim_3_kernel.complete_data(dataset=3, inplace=False)
sim_3_mice_comp_5 = sim_3_kernel.complete_data(dataset=4, inplace=False)
sim_3_mice_comp_6 = sim_3_kernel.complete_data(dataset=5, inplace=False)
sim_3_mice_comp_7 = sim_3_kernel.complete_data(dataset=6, inplace=False)
sim_3_mice_comp_8 = sim_3_kernel.complete_data(dataset=7, inplace=False)
sim_3_mice_comp_9 = sim_3_kernel.complete_data(dataset=8, inplace=False)
sim_3_mice_comp_10 = sim_3_kernel.complete_data(dataset=9, inplace=False)


sim_3_mar_drop = sim_3_mar.dropna(axis = 0, how = 'any', inplace = False)
sim_3_mar_drop = sim_3_mar_drop.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

fig = plt.figure()
plt.subplot(1,2,1)
sns.kdeplot(data = sim_3_comp, x = 'ColumnA', linestyle = 'solid', color = 'red')
sns.kdeplot(data = sim_3_mar_drop, x = 'ColumnA', linestyle = 'dashed', color = 'red')
sns.kdeplot(data = sim_3_mice_comp_1, x = 'ColumnA', linestyle = 'solid', color = '#e6e6e6')
sns.kdeplot(data = sim_3_mice_comp_2, x = 'ColumnA', linestyle = 'solid', color = '#d9d9d9')
sns.kdeplot(data = sim_3_mice_comp_3, x = 'ColumnA', linestyle = 'solid', color = '#bfbfbf')
sns.kdeplot(data = sim_3_mice_comp_4, x = 'ColumnA', linestyle = 'solid', color = '#b3b3b3')
sns.kdeplot(data = sim_3_mice_comp_5, x = 'ColumnA', linestyle = 'solid', color = '#a6a6a6')
sns.kdeplot(data = sim_3_mice_comp_6, x = 'ColumnA', linestyle = 'solid', color = '#999999')
sns.kdeplot(data = sim_3_mice_comp_7, x = 'ColumnA', linestyle = 'solid', color = '#8c8c8c')
sns.kdeplot(data = sim_3_mice_comp_8, x = 'ColumnA', linestyle = 'solid', color = '#808080')
sns.kdeplot(data = sim_3_mice_comp_9, x = 'ColumnA', linestyle = 'solid', color = '#404040')
sns.kdeplot(data = sim_3_mice_comp_10, x = 'ColumnA', linestyle = 'solid', color = '#000000')
plt.subplot(1,2,2)
sns.kdeplot(data = sim_3_comp, x = 'ColumnC', linestyle = 'solid', color = 'red')
sns.kdeplot(data = sim_3_mar_drop, x = 'ColumnC', linestyle = 'dashed', color = 'red')
sns.kdeplot(data = sim_3_mice_comp_1, x = 'ColumnC', linestyle = 'solid', color = '#e6e6e6')
sns.kdeplot(data = sim_3_mice_comp_2, x = 'ColumnC', linestyle = 'solid', color = '#d9d9d9')
sns.kdeplot(data = sim_3_mice_comp_3, x = 'ColumnC', linestyle = 'solid', color = '#bfbfbf')
sns.kdeplot(data = sim_3_mice_comp_4, x = 'ColumnC', linestyle = 'solid', color = '#b3b3b3')
sns.kdeplot(data = sim_3_mice_comp_5, x = 'ColumnC', linestyle = 'solid', color = '#a6a6a6')
sns.kdeplot(data = sim_3_mice_comp_6, x = 'ColumnC', linestyle = 'solid', color = '#999999')
sns.kdeplot(data = sim_3_mice_comp_7, x = 'ColumnC', linestyle = 'solid', color = '#8c8c8c')
sns.kdeplot(data = sim_3_mice_comp_8, x = 'ColumnC', linestyle = 'solid', color = '#808080')
sns.kdeplot(data = sim_3_mice_comp_9, x = 'ColumnC', linestyle = 'solid', color = '#404040')
sns.kdeplot(data = sim_3_mice_comp_10, x = 'ColumnC', linestyle = 'solid', color = '#000000')
fig.text(0, 0.1, 'Data Source: 3-Variable Simulated Data.\nSolid Red is the complete data.\nDashed Red is MAR data.\nLight grey to black represents first to last iteration.')
fig.subplots_adjust(bottom = 0.35)
plt.savefig('figures/sim_3_plot_imputed_dists.jpeg')

sim_3_kernel.plot_correlations(wspace = 0.5, hspace = 1.1)
plt.savefig('figures/sim_3_plot_correlations.jpeg')
sim_3_kernel.plot_mean_convergence(wspace = 0.5, hspace = 1.1)
plt.savefig('figures/sim_3_plot_mean_convergence.jpeg')

print('sim 3 complete')

	#* 10 Variable
sim_10_mar = sim_10_mar.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

sim_10_kernel = mf.ImputationKernel(
                                    sim_10_mar,
                                    datasets = 10,
                                    save_all_iterations = True,
                                    random_state = 601)
sim_10_kernel.mice(10)

sim_10_mice_comp_1 = sim_10_kernel.complete_data(dataset=0, inplace=False)
sim_10_mice_comp_2 = sim_10_kernel.complete_data(dataset=1, inplace=False)
sim_10_mice_comp_3 = sim_10_kernel.complete_data(dataset=2, inplace=False)
sim_10_mice_comp_4 = sim_10_kernel.complete_data(dataset=3, inplace=False)
sim_10_mice_comp_5 = sim_10_kernel.complete_data(dataset=4, inplace=False)
sim_10_mice_comp_6 = sim_10_kernel.complete_data(dataset=5, inplace=False)
sim_10_mice_comp_7 = sim_10_kernel.complete_data(dataset=6, inplace=False)
sim_10_mice_comp_8 = sim_10_kernel.complete_data(dataset=7, inplace=False)
sim_10_mice_comp_9 = sim_10_kernel.complete_data(dataset=8, inplace=False)
sim_10_mice_comp_10 = sim_10_kernel.complete_data(dataset=9, inplace=False)


sim_10_mar_drop = sim_10_mar.dropna(axis = 0, how = 'any', inplace = False)
sim_10_mar_drop = sim_10_mar_drop.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

fig = plt.figure()
plt.subplot(1,2,1)
sns.kdeplot(data = sim_10_comp, x = 'ColumnF', linestyle = 'solid', color = 'red')
sns.kdeplot(data = sim_10_mar_drop, x = 'ColumnF', linestyle = 'dashed', color = 'red')
sns.kdeplot(data = sim_10_mice_comp_1, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_2, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_3, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_4, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_5, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_6, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_7, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_8, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_9, x = 'ColumnF', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_10, x = 'ColumnF', linestyle = 'solid', color = 'black')
plt.subplot(1,2,2)
sns.kdeplot(data = sim_10_comp, x = 'ColumnC', linestyle = 'solid', color = 'red')
sns.kdeplot(data = sim_10_mar_drop, x = 'ColumnC', linestyle = 'dashed', color = 'red')
sns.kdeplot(data = sim_10_mice_comp_1, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_2, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_3, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_4, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_5, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_6, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_7, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_8, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_9, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_10_mice_comp_10, x = 'ColumnC', linestyle = 'solid', color = 'black')
fig.text(0, 0.1, 'Data Source: 10-Variable Simulated Data.\nSolid Red is the complete data.\nDashed Red is MAR data.\nLight grey to black represents first to last iteration.')
fig.subplots_adjust(bottom = 0.35)
plt.savefig('figures/sim_10_plot_imputed_dists.jpeg')

sim_10_kernel.plot_correlations(wspace = 0.5, hspace = 1.1)
plt.savefig('figures/sim_10_plot_correlations.jpeg')
sim_10_kernel.plot_mean_convergence(wspace = 0.5, hspace = 1.1)
plt.savefig('figures/sim_10_plot_mean_convergence.jpeg')

print('sim 10 complete')
	#* Performance
accuracy_3_a = sim_3_comp['ColumnA'] - sim_3_mice_comp_10['ColumnA']
accuracy_3_a = accuracy_3_a.mean()
accuracy_3_b = sim_3_comp['ColumnB'] - sim_3_mice_comp_10['ColumnB']
accuracy_3_b = accuracy_3_b.mean()
accuracy_3 = [('Column A', accuracy_3_a), ('Column B', accuracy_3_b)]
headers = ['Variable', 'Mean Difference']
print(tabulate(accuracy_3, headers, tablefmt = 'latex', floatfmt = '.3f'))
accuracy_10_b = sim_10_comp['ColumnB'] - sim_10_mice_comp_10['ColumnB']
accuracy_10_b = accuracy_10_b.mean()
accuracy_10_c = sim_10_comp['ColumnC'] - sim_10_mice_comp_10['ColumnC']
accuracy_10_c = accuracy_10_c.mean()
accuracy_10_f = sim_10_comp['ColumnF'] - sim_10_mice_comp_10['ColumnF']
accuracy_10_f = accuracy_10_f.mean()
accuracy_10_g = sim_10_comp['ColumnG'] - sim_10_mice_comp_10['ColumnG']
accuracy_10_g = accuracy_10_g.mean()
accuracy_10_h = sim_10_comp['ColumnH'] - sim_10_mice_comp_10['ColumnH']
accuracy_10_h = accuracy_10_h.mean()
accuracy_10_i = sim_10_comp['ColumnI'] - sim_10_mice_comp_10['ColumnI']
accuracy_10_i = accuracy_10_i.mean()
accuracy_10 = [('Column B', accuracy_10_b), ('Column C', accuracy_10_c), ('Column F', accuracy_10_f), ('Column G', accuracy_10_g), ('Column H', accuracy_10_h), ('Column I', accuracy_10_i)]
print(tabulate(accuracy_10, headers, tablefmt = 'latex', floatfmt = '.3f'))

print('performance complete')
# World Values Survey
	#* Full
wvs_mar = wvs_mar.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

wvs_kernel = mf.ImputationKernel(
                                   wvs_mar,
                                   datasets=1,
                                   save_all_iterations=True,
                                   random_state=601)
wvs_kernel.mice(iterations = 10, n_estimators = 5)

print('wvs kernel complete')

wvs_mice_comp_1 = wvs_kernel.complete_data(dataset=0, inplace=False)
wvs_mice_comp_2 = wvs_kernel.complete_data(dataset=1, inplace=False)
wvs_mice_comp_3 = wvs_kernel.complete_data(dataset=2, inplace=False)
wvs_mice_comp_4 = wvs_kernel.complete_data(dataset=3, inplace=False)
wvs_mice_comp_5 = wvs_kernel.complete_data(dataset=4, inplace=False)
wvs_mice_comp_6 = wvs_kernel.complete_data(dataset=5, inplace=False)
wvs_mice_comp_7 = wvs_kernel.complete_data(dataset=6, inplace=False)
wvs_mice_comp_8 = wvs_kernel.complete_data(dataset=7, inplace=False)
wvs_mice_comp_9 = wvs_kernel.complete_data(dataset=8, inplace=False)
wvs_mice_comp_10 = wvs_kernel.complete_data(dataset=9, inplace=False)


wvs_mar_drop = wvs_mar.dropna(axis = 0, how = 'any', inplace = False)
wvs_mar_drop = wvs_mar_drop.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


#fig = plt.figure()
#plt.subplot(1,2,1)
#sns.kdeplot(data = wvs_comp, x = 'ColumnF', linestyle = 'solid', color = 'red')
#sns.kdeplot(data = sim_10_mar_drop, x = 'ColumnF', linestyle = 'dashed', color = 'red')
#sns.kdeplot(data = sim_10_mice_comp_1, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_2, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_3, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_4, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_5, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_6, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_7, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_8, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_9, x = 'ColumnF', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_10, x = 'ColumnF', linestyle = 'solid', color = 'black')
#plt.subplot(1,2,2)
#sns.kdeplot(data = sim_10_comp, x = 'ColumnC', linestyle = 'solid', color = 'red')
#sns.kdeplot(data = sim_10_mar_drop, x = 'ColumnC', linestyle = 'dashed', color = 'red')
#sns.kdeplot(data = sim_10_mice_comp_1, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_2, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_3, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_4, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_5, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_6, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_7, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_8, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_9, x = 'ColumnC', linestyle = 'solid', color = 'black')
#sns.kdeplot(data = sim_10_mice_comp_10, x = 'ColumnC', linestyle = 'solid', color = 'black')
#fig.text(0, 0.1, 'Data Source: 10-Variable Simulated Data.\nSolid Red is the complete data.\nDashed Red is MAR data.\nLight grey to black represents first to last iteration.')
#fig.subplots_adjust(bottom = 0.35)
#plt.savefig('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/figures/sim_10_plot_imputed_dists.jpeg')

wvs_kernel.plot_correlations(wspace = 0.5, hspace = 1.1)
plt.savefig('figures/wvs_plot_correlations.jpeg')
wvs_kernel.plot_mean_convergence(wspace = 0.5, hspace = 1.1)
plt.savefig('figures/wvs_plot_mean_convergence.jpeg')
	#* Subset

#wvs_low_k_mar = wvs_low_k_mar.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

#wvs_low_k_kernel = mf.ImputationKernel(
#                                   wvs_low_k_mar,
#                                   datasets=10,
#                                   save_all_iterations=True,
#                                   random_state=601)
#wvs_low_k_kernel.mice(10)

#print('wvs low k kernel complete')
#wvs_low_k_mice_comp_1 = wvs_low_k_kernel.complete_data(dataset=0, inplace=False)
#wvs_low_k_mice_comp_2 = wvs_low_k_kernel.complete_data(dataset=1, inplace=False)
#wvs_low_k_mice_comp_3 = wvs_low_k_kernel.complete_data(dataset=2, inplace=False)
#wvs_low_k_mice_comp_4 = wvs_low_k_kernel.complete_data(dataset=3, inplace=False)
#wvs_low_k_mice_comp_5 = wvs_low_k_kernel.complete_data(dataset=4, inplace=False)
#wvs_low_k_mice_comp_6 = wvs_low_k_kernel.complete_data(dataset=5, inplace=False)
#wvs_low_k_mice_comp_7 = wvs_low_k_kernel.complete_data(dataset=6, inplace=False)
#wvs_low_k_mice_comp_8 = wvs_low_k_kernel.complete_data(dataset=7, inplace=False)
#wvs_low_k_mice_comp_9 = wvs_low_k_kernel.complete_data(dataset=8, inplace=False)
#wvs_low_k_mice_comp_10 = wvs_low_k_kernel.complete_data(dataset=9, inplace=False)


#wvs_low_k_mar_drop = wvs_low_k_mar.dropna(axis = 0, how = 'any', inplace = False)
#wvs_low_k_mar_drop = wvs_low_k_mar_drop.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))