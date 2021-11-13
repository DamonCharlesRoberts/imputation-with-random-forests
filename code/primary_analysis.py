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
from plotnine import ggplot, aes, geom_density
import re
	#* Plot Style 
sns.set(style = 'whitegrid')
# Simulated Data 
	#* Import complete data
sim_3_comp = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_complete.csv')
sim_3_comp = sim_3_comp.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
sim_10_comp = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_complete.csv')
sim_10_comp = sim_10_comp.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
	#* Import MAR data
sim_3_mar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_MAR.csv')
sim_10_mar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_MAR.csv')
	#* Import MCAR data
#sim_3_mcar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_MCAR.csv')
#sim_10_mcar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_MCAR.csv')
	#* Import MNAR data
#sim_3_mnar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_three_MNAR.csv')
#sim_10_mnar = pd.read_csv(filepath_or_buffer = '/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/data/sim_ten_MNAR.csv')

# 3 variable

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


plt.subplot(1,2,1)
sns.kdeplot(data = sim_3_comp, x = 'ColumnA', linestyle = 'solid', color = 'red')
sns.kdeplot(data = sim_3_mar_drop, x = 'ColumnA', linestyle = 'dashed', color = 'red')
sns.kdeplot(data = sim_3_mice_comp_1, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_2, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_3, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_4, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_5, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_6, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_7, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_8, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_9, x = 'ColumnA', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_10, x = 'ColumnA', linestyle = 'solid', color = 'black')
plt.subplot(1,2,2)
sns.kdeplot(data = sim_3_comp, x = 'ColumnC', linestyle = 'solid', color = 'red')
sns.kdeplot(data = sim_3_mar_drop, x = 'ColumnC', linestyle = 'dashed', color = 'red')
sns.kdeplot(data = sim_3_mice_comp_1, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_2, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_3, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_4, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_5, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_6, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_7, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_8, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_9, x = 'ColumnC', linestyle = 'solid', color = 'black')
sns.kdeplot(data = sim_3_mice_comp_10, x = 'ColumnC', linestyle = 'solid', color = 'black')
plt.tight_layout()
plt.savefig('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/figures/sim_3_plot_imputed_dists.jpeg')

sim_3_kernel.plot_correlations(wspace = 0.5, hspace = 1.1)
plt.savefig('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/figures/sim_3_plot_correlations.jpeg')
sim_3_kernel.plot_mean_convergence(wspace = 0.5, hspace = 1.1)
plt.savefig('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/figures/sim_3_plot_mean_convergence.jpeg')
# 10 Variable
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
plt.tight_layout()
plt.savefig('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/figures/sim_10_plot_imputed_dists.jpeg')

sim_10_kernel.plot_correlations(wspace = 0.5, hspace = 1.1)
plt.savefig('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/figures/sim_10_plot_correlations.jpeg')
sim_10_kernel.plot_mean_convergence(wspace = 0.5, hspace = 1.1)
plt.savefig('/Users/damonroberts/Dropbox/current_projects/dcr_rf_imputation/figures/sim_10_plot_mean_convergence.jpeg')