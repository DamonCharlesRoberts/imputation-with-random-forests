<h4 align="center">Replication Code: Connecting Leaves to the Forest</h4>
<p align="center">
    <a href="https://github.com/DamonCharlesRoberts/imputation-with-random-forests/commits/main">
    <img src="https://img.shields.io/github/last-commit/DamonCharlesRoberts/mputation-with-random-forests.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub last commit"></a>
    <a href="https://github.com/DamonCharlesRoberts/mputation-with-random-forests/issues">
    <img src="https://img.shields.io/github/issues-raw/DamonCharlesRoberts/mputation-with-random-forests.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub issues"></a>
    <a href="https://github.com/DamonCharlesRoberts/mputation-with-random-forests/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/DamonCharlesRoberts/imputation-with-random-forests.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub pull requests"></a>
</p>

--- 
Academic project examining the utility of using Random Forest models for multiple imputation with chained equations in political science. 



# Code

* `produce_na_function.py`: A function for generating MCAR, MAR, and MNAR missigness to data. 
* `missingness.py`: Generating simulated data, executing function to generate MCAR, MAR, and MNAR missingness to simulated data.
* `rmse_function.py`: Defines a function that calculates RMSE for `simulation_imputation_and_performance.py` script.
* `simulation_imputation_and_performance.py`: Imputing missing values using `miceforest` library. Checks performance on imputing simulated data with MAR.
* `wvs_imputation.py`: Imputing missing values using `miceforest` library in World Values Survey Data.
* `wvs_performance.py`: Runs regressions on voter turnout using imputed datasets and LWD for the WVS. Computes t-test to compare coefficients between the models. 
