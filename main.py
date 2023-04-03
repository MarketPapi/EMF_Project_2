from pathlib import Path
from scripts.parameters import paths
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scripts.functions as fn
import statsmodels.api as sm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# %%
# **************************************************
# *** QUESTION 1: Stationarity                   ***
# **************************************************

# Importing dataset (commodities)
df_data = fn.read_data(file_path=Path.joinpath(paths.get('data'), 'commodities.csv'))

# Taking logarithm of the adjusted close price
l_adj_close_price = ['ZC Adj Close', 'ZW Adj Close', 'ZS Adj Close', 'KC Adj Close', 'CC Adj Close']
df_data = df_data[l_adj_close_price]
df_data_ln = fn.log_transform_cols(df_data, l_adj_close_price)


# %%
# **************************************************
# *** QUESTION 1.1: Critical Values              ***
# **************************************************

# Initialisation
T = len(df_data_ln.index)
N = 10000
column = 'ZC Adj Close'

# First simulation
simulation_1 = fn.critical_value(df_data_ln, column, T, N)
ar_params_1 = simulation_1[1]

# *** Question 1.4 ***
# Plot the histogram of the Test-Statistic
fig, ax = plt.subplots(figsize=(15, 10))
ax.hist(ar_params_1['DF_TS'], bins=50, edgecolor='black')
ax.set_title('Test Statistic Distribution for N: ' + str(N))
plt.xlabel('Test Statistic')
plt.ylabel('Frequency')
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q1.4_T-Stat Distribution.png'))
plt.close()

# *** Question 1.5 ***
# Compute the critical values of the DF test
critical_values_1 = simulation_1[0]
critical_values_1.to_latex(Path.joinpath(paths.get('output'), 'Q1.5_Critical_Values.tex'), float_format='%.2f')

# Plot the histogram of the Test-Statistic and Critical Values
fig, ax = plt.subplots(figsize=(15, 10))
ax.hist(ar_params_1['DF_TS'], bins=50, edgecolor='black')
ax.set_title('Test Statistic Distribution for N: ' + str(N))
plt.xlabel('Test Statistic')
plt.ylabel('Frequency')
plt.axvline(x=critical_values_1.loc[0.01], color='r', label='CV 1%')
plt.axvline(x=critical_values_1.loc[0.05], color='y', label='CV 5%')
plt.axvline(x=critical_values_1.loc[0.10], color='g', label='CV 10%')
plt.legend()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q1.5_T-Stat Distribution_CV.png'))
plt.close()

# *** Question 1.6 ***
# Re-computing the simulation for T=500
simulation_2 = fn.critical_value(df_data_ln, column, T=500, N=N)
critical_values_2 = simulation_2[0]


# %%
# **************************************************
# *** QUESTION 1.2: Testing Non-Stationarity     ***
# **************************************************

# *** Question 1.7 ***
# Computing the DF Test
DF_Test = pd.DataFrame(index=['DF_TS', 'CV 1%', 'CV 5%', 'CV 10%', 'Reject H0 1%', 'Reject H0 5%', 'Reject H0 10%',
                              'P_Value'], columns=l_adj_close_price)

for col in l_adj_close_price:
    t_stat_data = fn.reg(df_data_ln, col, lag=1)
    DF_Test.loc['DF_TS'][col] = t_stat_data
    DF_Test.loc['CV 1%'][col] = critical_values_1.loc[0.01]
    DF_Test.loc['CV 5%'][col] = critical_values_1.loc[0.05]
    DF_Test.loc['CV 10%'][col] = critical_values_1.loc[0.10]
    DF_Test.loc['Reject H0 1%'][col] = np.abs(DF_Test.loc['DF_TS'][col]) > np.abs(DF_Test.loc['CV 1%'][col])
    DF_Test.loc['Reject H0 5%'][col] = np.abs(DF_Test.loc['DF_TS'][col]) > np.abs(DF_Test.loc['CV 5%'][col])
    DF_Test.loc['Reject H0 10%'][col] = np.abs(DF_Test.loc['DF_TS'][col]) > np.abs(DF_Test.loc['CV 10%'][col])
    DF_Test.loc['P_Value'][col] = fn.get_pvalue(ar_params_1['DF_TS'], DF_Test.loc['DF_TS'][col])

DF_Test.columns = ['Corn', 'Wheat', 'Soybean', 'Coffee', 'Cacao']
DF_Test = fn.format_float(DF_Test)
DF_Test.to_latex(Path.joinpath(paths.get('output'), 'Q1.7_DF_Test.tex'))


# %%
# **************************************************
# *** QUESTION 2: Cointegration                  ***
# **************************************************


# %%
# **************************************************
# *** QUESTION 2.1: Critical Values              ***
# **************************************************

# Compute critical values
t_stat_coint = fn.simulate_coint_cv(T, N)
# Second simulation for later use.
t_stat_coint_500 = fn.simulate_coint_cv(T=500, N=N)
df_ts_coint = pd.DataFrame(data=t_stat_coint, columns=['DF_TS'])
cv_coint = df_ts_coint['DF_TS'].quantile([0.01, 0.05, 0.1])
cv_coint = cv_coint.rename('Critical Value')
# Save as Latex
cv_coint.to_latex(Path.joinpath(paths.get('output'), 'Q2.1_Critical_Values_Coint.tex'), float_format='%.2f')


# Plotting Histogram of critical values.
fig, ax = plt.subplots(figsize=(15, 10))
ax.hist(df_ts_coint['DF_TS'], bins=50, edgecolor='black')
ax.set_title('Test Statistic Distribution for N: ' + str(N))
plt.xlabel('Test Statistic')
plt.ylabel('Frequency')
plt.axvline(x=cv_coint.loc[0.01], color='r', label='CV 1%')
plt.axvline(x=cv_coint.loc[0.05], color='y', label='CV 5%')
plt.axvline(x=cv_coint.loc[0.10], color='g', label='CV 10%')
plt.legend()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q2.1_T-Stat_Distribution_Coint.png'))
plt.close()


# %%
# **************************************************
# *** QUESTION 2.2: Testing for Cointegration    ***
# **************************************************

# Cointegration results DataFrame, needed to construct another cointegration test statistics DataFrame
df_coint = fn.cointgration(df_data_ln)

# *** Question 2.2 ***
coint_test = pd.DataFrame(index=df_coint.index,
                          columns=['DF_TS', 'CV_1%', 'CV_5%', 'CV_10%', 'P_Value', 'Reject H0 1%', 'Reject H0 5%',
                                   'Reject H0 10%'])

for index in df_coint.index:
    coint_test.loc[index]['DF_TS'] = df_coint.loc[index]['DF_TS']
    coint_test.loc[index]['CV_1%'] = cv_coint.loc[0.01]
    coint_test.loc[index]['CV_5%'] = cv_coint.loc[0.05]
    coint_test.loc[index]['CV_10%'] = cv_coint.loc[0.1]
    coint_test.loc[index]['P_Value'] = fn.get_pvalue(t_stat_coint, df_coint.loc[index]['DF_TS'])
    coint_test.loc[index]['Reject H0 1%'] = np.abs(df_coint.loc[index]['DF_TS']) > np.abs(cv_coint.loc[0.01])
    coint_test.loc[index]['Reject H0 5%'] = np.abs(df_coint.loc[index]['DF_TS']) > np.abs(cv_coint.loc[0.05])
    coint_test.loc[index]['Reject H0 10%'] = np.abs(df_coint.loc[index]['DF_TS']) > np.abs(cv_coint.loc[0.1])

coint_test_out = fn.format_float(coint_test)
coint_test_out.to_latex(Path.joinpath(paths.get('output'), 'Q2.2_Coint_Test_Results.tex'))

# *** Question 2.3 ***
df_coint_out = fn.format_float(df_coint[['Alpha', 'Beta']])
df_coint_out.to_latex(Path.joinpath(paths.get('output'), 'Q2.3_A_B_Values.tex'))

# *** Question 2.5 ***
pA = df_data_ln['ZW Adj Close']
alpha = df_coint.loc['Wheat-Corn']['Alpha']
beta = df_coint.loc['Wheat-Corn']['Beta']
pB = df_data_ln['ZC Adj Close']
comb = alpha + beta * pB

# Plot PA and Linear combination
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pd.to_datetime(pA.index), pA, label='Wheat', c='blue')
ax.plot(pd.to_datetime(pA.index), comb, label='alpha + beta * Price B (Corn)', c='green')
year_locator = mdates.YearLocator()
year_formatter = mdates.DateFormatter('%Y')
ax.xaxis.set_major_locator(year_locator)
ax.xaxis.set_major_formatter(year_formatter)
ax.set_xlabel('Time')
ax.set_ylabel('Log Price')
ax.set_title('Wheat-Corn Pair')
ax.legend()
plt.show()
fig.autofmt_xdate()
fig.savefig(Path.joinpath(paths.get('output'), 'Q2.5_WC_Pair_Plot.png'))
plt.close()


# %%
# **************************************************
# *** QUESTION 3: Pair Trading                   ***
# **************************************************

# Importing dataset (commodities)
df_data = fn.read_data(file_path=Path.joinpath(paths.get('data'), 'commodities.csv'))
l_adj_close_price = ['ZC Adj Close', 'ZW Adj Close', 'ZS Adj Close', 'KC Adj Close', 'CC Adj Close']
df_data = df_data[l_adj_close_price]


# %%
# **************************************************
# *** QUESTION 3.1: Trading Signal               ***
# **************************************************

# *** Question 3.1 ***
"""
Model: P_t^{A} = a + b*P_t^{B} + z_t
Conclusion: as long as cointegration relation holds, we should expect z_t = 0, i.e. no spread
Arbitrage: z_t >> 0 ==> P_t^{A} - (a + b*P_t^{B}) >> 0 we have that the price of A is significantly above the cointegrated
price of A ==> we expect P_t^{A} to back to the cointegrated price, so we short A and use the proceeds to long B, and we
make a profit as long as the two prices converge back to cointegrated price
Strategy: z_t >> 0 (spread >> 0) ==> short A, long B; z_t << 0 (spread << 0) ==> short B, long A
"""

# *** Question 3.2 ***
# Best (cointegrated) pair: A=Wheat (ZW Adj Close), B=Corn (ZC Adj Close) ==> reg Corn (X) on Wheat (y)
# Concept: we will use spreads to create trading signals
s_spreads = fn.tab_spreads(df_data=df_data, A='ZW Adj Close', B='ZC Adj Close')

# Plot spreads
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pd.to_datetime(s_spreads.index), s_spreads, label='Spread', c='red')
year_locator = mdates.YearLocator()
year_formatter = mdates.DateFormatter('%Y')
ax.xaxis.set_major_locator(year_locator)
ax.xaxis.set_major_formatter(year_formatter)
ax.set_title('Wheat-Corn Pair')
ax.legend()
plt.show()
fig.autofmt_xdate()
fig.savefig(Path.joinpath(paths.get('output'), 'Q3.2_Spreads.png'))
plt.close()


# *** Question 3.3 ***
# Autocorrelogram of spreads
fn.plot_autocorrelogram(s_data=s_spreads, outfile=Path.joinpath(paths.get('output'), 'Q3.3_Autocorrelogram_Spreads.png'))

# Ljung-Box test with p=10 lags
fn.Ljung_Box_test(s_data=s_spreads)

"""
Observation: both autocorrelogram and Ljung-Box test indicate that spread process z_t^{tilde} is autocorrelated ==>
we reject null rho_1 = ... = rho_p = 0 with p=10 lags at virtually any confidence level, and in correlogram we see that
autocorrelations rho_k for k = 1, ..., 10 are all above the confidence interval @95%, hence rho_k are all significantly
different from zero
Implication: ???
"""


# %%
# **************************************************
# *** QUESTION 3.2: In-Sample Pair Trading       ***
# **************************************************


# %%
# **************************************************
# *** QUESTION 3.2.1: Direct Strategy            ***
# **************************************************

# *** Question 3.4 ***
def tab_PT_insample(df_data, A, B, W=1000, L=2, in_level=1.5, stop_level=None):
    # Assumption: sig1 open @t=0 and sig1 close @t=1 ==> open position at close t=0, and close position at close t=1
    # Initialization
    df_PT_insample = pd.DataFrame()
    df_PT_insample[A] = df_data[A]
    df_PT_insample[B] = df_data[B]
    df_PT_insample['Spread'] = fn.tab_spreads(df_data=df_data, A=A, B=B)

    # Signals
    df_PT_insample['Sig1 Open'] = (df_PT_insample['Spread'] > in_level)
    df_PT_insample['Sig2 Open'] = (df_PT_insample['Spread'] < -in_level)
    df_PT_insample['Sig1 Close'] = (df_PT_insample['Spread'] <= 0)
    df_PT_insample['Sig2 Close'] = (df_PT_insample['Spread'] >= 0)
    if stop_level is not None:
        df_PT_insample['Sig1 Stop'] = (df_PT_insample['Spread'] > stop_level)
        df_PT_insample['Sig2 Stop'] = (df_PT_insample['Spread'] < -stop_level)

    # Positions
    curr = df_PT_insample.index[0]
    df_PT_insample.loc[curr, 'Pos1'] = df_PT_insample.loc[curr, 'Sig1 Open']
    df_PT_insample.loc[curr, 'Pos2'] = df_PT_insample.loc[curr, 'Sig2 Open']
    for i in range(1, len(df_PT_insample.index[1:])+1):
        prev = df_PT_insample.index[i-1]
        curr = df_PT_insample.index[i]
        if stop_level is None:
            df_PT_insample.loc[curr, 'Pos1'] = (df_PT_insample.loc[curr, 'Sig1 Open'] or (df_PT_insample.loc[prev, 'Pos1'] and not df_PT_insample.loc[prev, 'Sig1 Close']))
            df_PT_insample.loc[curr, 'Pos2'] = (df_PT_insample.loc[curr, 'Sig2 Open'] or (df_PT_insample.loc[prev, 'Pos2'] and not df_PT_insample.loc[prev, 'Sig2 Close']))
        elif stop_level is not None:
            df_PT_insample.loc[curr, 'Pos1'] = (df_PT_insample.loc[curr, 'Sig1 Open'] or (df_PT_insample.loc[prev, 'Pos1'] and not df_PT_insample.loc[prev, 'Sig1 Close'] and not df_PT_insample.loc[prev, 'Sig1 Stop']))
            df_PT_insample.loc[curr, 'Pos2'] = (df_PT_insample.loc[curr, 'Sig2 Open'] or (df_PT_insample.loc[prev, 'Pos2'] and not df_PT_insample.loc[prev, 'Sig2 Close'] and not df_PT_insample.loc[prev, 'Sig2 Stop']))
    for col in df_PT_insample.columns[3:]:
        df_PT_insample[col] = df_PT_insample[col].astype('bool')

    return df_PT_insample


df_test = tab_PT_insample(df_data=df_data, A='ZW Adj Close', B='ZC Adj Close', stop_level=1.75)
# *** Question 3.5 ***

















# %%
# **************************************************
# *** QUESTION 3.2.2: Stop Loss                  ***
# **************************************************

# *** Question 3.6 ***
# *** Question 3.7 ***


# %%
# **************************************************
# *** QUESTION 3.3: Out-of-Sample Pair Trading   ***
# **************************************************

# *** Question 3.8 ***
# *** Question 3.9 ***
# *** Question 3.10 ***
# *** Question 3.11 ***
# *** Question 3.12 ***
# *** Question 3.13 ***