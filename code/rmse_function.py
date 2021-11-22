# Title: RMSE Function

# Notes:
	# Description: Script that defines `rmse` function
	# Updated: 2021-11-22
	# Updated by: dcr


def rmse(original, imputed):
	differences = original - imputed
	differences_squared = differences ** 2
	mean_of_difference_squared = differences_squared.mean()
	rmse_val = np.sqrt(mean_of_difference_squared)
	rmse_val = pd.DataFrame(rmse_val)
	rmse_val['rmse'] = rmse_val.iloc[:,0]
	rmse_val= rmse_val[['rmse']].mean()
	return rmse_val