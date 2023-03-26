import matplotlib.pyplot as plt
import pandas as pd
import scripts.project_functions as pf
from pathlib import Path
from scripts.project_parameters import paths
import numpy as np
import warnings


# Ignore warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('fivethirtyeight')

# Importing the commodities data
df_data = pf.read_data(file_path=Path.joinpath(paths.get('data'), 'commodities.csv'))

# Taking Logarithm of the adjusted close price
l_adj_close_price = ['ZC Adj Close', 'ZW Adj Close', 'ZS Adj Close', 'KC Adj Close', 'CC Adj Close']
df_data_ln = pf.log_transform_cols(df_data, l_adj_close_price)

# %%
# **************************************************
# *** 1: Stationarity                            ***
# **************************************************

# *** Question 1.1  ***


# *** Question 1.2  ***#


# %%
# **************************************************
# *** 1.1: Critical Values              ***
# **************************************************


# Initialisation
T = len(df_data_ln.index)
N = 10000
column = 'ZC Adj Close'

# First Simulation
simulation_1 = pf.critical_value(df_data_ln, column, T, N)
ar_params_1 = simulation_1[1]

# *** Question 1.4 ***
# Plot the histogram of the Test-Statistic.
fig, ax = plt.subplots(figsize=(15, 10))
ax.hist(ar_params_1['DF_TS'], bins=50)
ax.set_title('Critical Value distribution for N: ' + str(N))
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q1.4_Critical_Value_Histogram.png'))
plt.close()

# *** Question 1.5 ***
# Compute the critical values of the DF test.
critical_values_1 = simulation_1[0]
critical_values_1.to_latex(Path.joinpath(paths.get('output'), 'Q1.5_Critical_Values.tex'), float_format="%.2f")

# *** Question 1.6 ***
# Re-computing the simulation for T=500.
simulation_2 = pf.critical_value(df_data_ln, column, T=500, N=N)
critical_values_2 = simulation_2[0]

# %%
# **************************************************
# *** 1.2: Testing Non-stationarity     ***
# **************************************************

# *** Question 1.7 ***
# Computing the DF Test.
DF_Test = pd.DataFrame(index=['DF_TS', 'CV 1%', 'CV 5%', 'CV 10%', 'Reject H0 1%', 'Reject H0 5%', 'Reject H0 10%'],
                       columns=l_adj_close_price)

for col in l_adj_close_price:
    t_stat_data = pf.reg(df_data_ln, col, lag=1)

    DF_Test.loc['DF_TS'][col] = t_stat_data
    DF_Test.loc['CV 1%'][col] = critical_values_1.loc[0.01]
    DF_Test.loc['CV 5%'][col] = critical_values_1.loc[0.05]
    DF_Test.loc['CV 10%'][col] = critical_values_1.loc[0.10]
    DF_Test.loc['Reject H0 1%'][col] = np.abs(DF_Test.loc['DF_TS'][col]) > np.abs(DF_Test.loc['CV 1%'][col])
    DF_Test.loc['Reject H0 5%'][col] = np.abs(DF_Test.loc['DF_TS'][col]) > np.abs(DF_Test.loc['CV 5%'][col])
    DF_Test.loc['Reject H0 10%'][col] = np.abs(DF_Test.loc['DF_TS'][col]) > np.abs(DF_Test.loc['CV 10%'][col])
    # TODO: Compute the p-value of the distribution

DF_Test.columns = ['ZC', 'ZW', 'ZS', 'KC', 'CC']

# *** Question 1.8 ***


# %%
# **************************************************
# *** 2: Cointegration                  ***
# **************************************************


# %%
# **************************************************
# *** 2.1: Critical Values             ***
# **************************************************

# *** Question 2.1 ***

# %%
# **************************************************
# *** 2.2: Testing for Cointegration             ***
# **************************************************

# *** Question 2.2 ***


# *** Question 2.3 ***


# *** Question 2.4 ***

# *** Question 2.5 ***


# %%
# **************************************************
# *** 3.1: Trading Signal                        ***
# **************************************************


# *** Question 3.1 ***

# *** Question 3.2 ***

# *** Question 3.3 ***


# %%
# **************************************************
# *** 3.2: In-sample Pair trading Strategy       ***
# **************************************************

# *** Question 3.4 ***

# *** Question 3.5 ***

# %%
# **************************************************
# *** 3.2.2: Stop Loss       ***
# **************************************************

# *** Question 3.6 ***

# *** Question 3.7 ***


# %%
# **************************************************
# *** 3.3: Out-of-sample Pair-trading Strategy   ***
# **************************************************

# *** Question 3.8 ***

# *** Question 3.9 ***

# *** Question 3.10 ***

# *** Question 3.11 ***

# *** Question 3.12 ***

# *** Question 3.13 ***
