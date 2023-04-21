# Import packages
from pathlib import Path
from scipy.stats import norm
from scripts.parameters import paths
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scripts.functions as fn
import seaborn as sns
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
sns.set(context='paper', style='ticks', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.histplot(data=ar_params_1, x='DF_TS', bins=50, edgecolor='black')
ax.set_title('DF T-Stat Distribution (N={})'.format(N), size=28)
ax.tick_params(axis='both', labelsize=18)
plt.xlabel('Test Statistic', size=20)
plt.ylabel('Frequency', size=20)
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q1.4_DF_tstat_dist.png'))
plt.close()

# *** Question 1.5 ***
# Compute the critical values of the DF test
critical_values_1 = simulation_1[0]
critical_values_1.to_latex(Path.joinpath(paths.get('output'), 'Q1.5_DF_cv.tex'), float_format='%.2f')

# Plot the histogram of the Test-Statistic and Critical Values
sns.set(context='paper', style='ticks', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.histplot(data=ar_params_1, x='DF_TS', bins=50, edgecolor='black')
plt.axvline(x=critical_values_1.loc[0.01], label='CV 1%', color='red', lw=3)
plt.axvline(x=critical_values_1.loc[0.05], label='CV 5%', color='orange', lw=3)
plt.axvline(x=critical_values_1.loc[0.10], label='CV 10%', color='green', lw=3)
ax.set_title('DF T-Stat Distribution with CV (N={})'.format(N), size=28)
ax.tick_params(axis='both', labelsize=18)
plt.xlabel('Test Statistic', size=20)
plt.ylabel('Frequency', size=20)
plt.legend()
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q1.5_DF_tstat_dist_cv.png'))
plt.close()

# *** Question 1.6 ***
# Re-computing the simulation for T=500
simulation_2 = fn.critical_value(df_data_ln, column, T=500, N=N)
critical_values_2 = simulation_2[0]


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
DF_Test.to_latex(Path.joinpath(paths.get('output'), 'Q1.7_DF_test_results.tex'))


# %%
# **************************************************
# *** QUESTION 2: Cointegration                  ***
# **************************************************


# **************************************************
# *** QUESTION 2.1: Critical Values              ***
# **************************************************

# Simulate distribution
t_stat_coint = fn.simulate_coint_cv(T, N=N)
df_ts_coint = pd.DataFrame(data=t_stat_coint, columns=['DF_TS'])

# Compute critical values
cv_coint = df_ts_coint['DF_TS'].quantile([0.01, 0.05, 0.1])
cv_coint = cv_coint.rename('Critical Value')
# Save as Latex
cv_coint.to_latex(Path.joinpath(paths.get('output'), 'Q2.1_coint_cv.tex'), float_format='%.2f')

# Plotting histogram of critical values
sns.set(context='paper', style='ticks', font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
sns.histplot(data=df_ts_coint, x='DF_TS', bins=50, edgecolor='black')
plt.axvline(x=cv_coint.loc[0.01], label='CV 1%', color='red', lw=3)
plt.axvline(x=cv_coint.loc[0.05], label='CV 5%', color='orange', lw=3)
plt.axvline(x=cv_coint.loc[0.10], label='CV 10%', color='green', lw=3)
ax.set_title('Cointegration T-Stat Distribution with CV (N={})'.format(N), size=28)
ax.tick_params(axis='both', labelsize=18)
plt.xlabel('Test Statistic', size=20)
plt.ylabel('Frequency', size=20)
plt.legend()
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q2.1_coint_tstat_dist_cv.png'))
plt.close()


# **************************************************
# *** QUESTION 2.2: Testing for Cointegration    ***
# **************************************************

# Cointegration results DataFrame, needed to construct another cointegration test statistics DataFrame
df_coint = fn.cointegration(df_data_ln, column_names=['Corn', 'Wheat', 'Soybean', 'Coffee', 'Cacao'], permut=True)

# *** Question 2.2 ***
coint_test = pd.DataFrame(index=df_coint.index, columns=['DF_TS', 'CV_1%', 'CV_5%', 'CV_10%', 'P_Value', 'Reject H0 1%', 'Reject H0 5%', 'Reject H0 10%'])

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
coint_test_out.to_latex(Path.joinpath(paths.get('output'), 'Q2.2_coint_test_results.tex'))
small_coint_test = coint_test_out[(coint_test_out['Reject H0 1%'] == True) | (coint_test_out['Reject H0 5%'] == True) | (coint_test_out['Reject H0 10%'] == True)]
small_coint_test.to_latex(Path.joinpath(paths.get('output'), 'Q2.2_small_coint_test_results.tex'))

# *** Question 2.3 ***
df_coint_out = fn.format_float(df_coint[['Alpha', 'Beta']])
df_coint_out.to_latex(Path.joinpath(paths.get('output'), 'Q2.3_AB_values.tex'))

# *** Question 2.5 ***
pA = df_data_ln['ZW Adj Close']
alpha = df_coint.loc['Wheat-Corn']['Alpha']
beta = df_coint.loc['Wheat-Corn']['Beta']
pB = df_data_ln['ZC Adj Close']
comb = alpha + beta * pB

# Plot PA and Linear combination
sns.set(context='paper', style='ticks', font_scale=1.0)
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot()
ax.grid(False)
ax.set_title(label='Wheat-Corn Pair', size=28)
# Items
sns.lineplot(x=pd.to_datetime(pA.index), y=pA, label='Wheat', color='sienna', lw=3)
sns.lineplot(x=pd.to_datetime(comb.index), y=comb, label='Alpha + Beta * PriceB (Corn)', color='gold', lw=3)
# X-axis settings
date_locator = mdates.YearLocator()
date_formatter = mdates.DateFormatter('%Y')
ax.tick_params(axis='x', labelrotation=0, labelsize=18)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(date_formatter)
ax.set_xlabel(xlabel='')
# Y-axis settings
ax.set_yticklabels(labels=['{:.1f}'.format(y) for y in ax.get_yticks()], size=18)
ax.set_ylabel(ylabel='Log Price', size=20)
# Legend settings
ax.legend(loc='upper left', fontsize=16)
# Show and save
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q2.5_WC_pair_plot.png'))
plt.close()


# %%
# **************************************************
# *** QUESTION 3: Pair Trading                   ***
# **************************************************

# Importing dataset (commodities)
df_data = fn.read_data(file_path=Path.joinpath(paths.get('data'), 'commodities.csv'))
l_adj_close_price = ['ZC Adj Close', 'ZW Adj Close', 'ZS Adj Close', 'KC Adj Close', 'CC Adj Close']
df_data = df_data[l_adj_close_price]

# Simulating distribution (cointegration)
t_stat_coint_500 = fn.simulate_coint_cv(T=500, N=10000)
df_ts_coint_500 = pd.DataFrame(data=t_stat_coint_500, columns=['DF_TS'])


# **************************************************
# *** QUESTION 3.1: Trading Signal               ***
# **************************************************

# Best pair: A=Wheat (ZW Adj Close), B=Corn (ZC Adj Close) ==> reg Corn (X) on Wheat (y)
# Concept: we will use spread to create trading signals

# *** Question 3.2 ***
# Compute spread
X = df_data[['ZC Adj Close']]
y = df_data['ZW Adj Close']
lr_model = LinearRegression()
lr_model.fit(X, y)
s_spread = y - lr_model.predict(X)
s_spread = s_spread / s_spread.std(ddof=0)

# Plot spread
sns.set(context='paper', style='ticks', font_scale=1.0)
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot()
ax.grid(False)
ax.set_title(label='Wheat-Corn Spread', size=28)
# Items
ax.axhline(y=0, color='black', ls='--', lw=2)
ax.axhline(y=1.5, color='green', ls='--', lw=2)
ax.axhline(y=-1.5, color='green', ls='--', lw=2)
ax.axhline(y=2.75, color='red', ls='--', lw=2)
ax.axhline(y=-2.75, color='red', ls='--', lw=2)
sns.lineplot(x=pd.to_datetime(s_spread.index), y=s_spread, label='Spread', color='lightcoral', lw=3)
# X-axis settings
date_locator = mdates.YearLocator()
date_formatter = mdates.DateFormatter('%Y')
ax.tick_params(axis='x', labelrotation=0, labelsize=18)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(date_formatter)
ax.set_xlabel(xlabel='')
# Y-axis settings
ax.set_yticklabels(labels=['{:.1f}'.format(y) for y in ax.get_yticks()], size=18)
ax.set_ylabel(ylabel='')
# Legend settings
ax.legend(loc='upper left', fontsize=16)
# Show and save
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q3.2_spread.png'))
plt.close()

# *** Question 3.3 ***
# Compute autocorrelogram of spread
df_autocorrelogram = fn.tab_autocorrelogram(s_data=s_spread, alpha=0.05, max_lags=10)

# Plot autocorrelogram of spread
sns.set(context='paper', style='ticks', font_scale=1.0)
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot()
ax.grid(False)
ax.set_title(label='Autocorrelogram Spread ({})'.format(df_autocorrelogram.columns[-1]), size=28)
# Items
s_autocorr = df_autocorrelogram['Autocorrelation']
s_ci_lower = pd.Series([df_autocorrelogram.iloc[:, -1][i][0] for i in df_autocorrelogram.index], index=df_autocorrelogram.index)
s_ci_upper = pd.Series([df_autocorrelogram.iloc[:, -1][i][1] for i in df_autocorrelogram.index], index=df_autocorrelogram.index)
ax.axhline(y=0, color='red', lw=3)
sns.lineplot(x=df_autocorrelogram.index, y=s_autocorr, color='black', lw=3)
sns.lineplot(x=df_autocorrelogram.index, y=s_ci_lower, color='blue', lw=3)
sns.lineplot(x=df_autocorrelogram.index, y=s_ci_upper, color='blue', lw=3)
# X-axis settings
ax.set_xticks(np.arange(df_autocorrelogram.index[0], df_autocorrelogram.index[-1]+1, 1))
ax.set_xlim(df_autocorrelogram.index[0], df_autocorrelogram.index[-1])
ax.set_xticklabels(labels=['{:.0f}'.format(x) for x in ax.get_xticks()], size=18)
ax.set_xlabel(xlabel='Lags', size=20)
# Y-axis settings
ax.set_yticklabels(labels=['{:.1f}'.format(y) for y in ax.get_yticks()], size=18)
ax.set_ylabel(ylabel='k-Lags Autocorrelation', size=20)
# Show and save
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q3.3_autocorrelogram_spread.png'))
plt.close()

# Ljung-Box test with p=10 lags
fn.Ljung_Box_test(s_data=s_spread)


# **************************************************
# *** QUESTION 3.2: In-Sample Pair Trading       ***
# **************************************************


# **************************************************
# *** QUESTION 3.2.1: Direct Strategy            ***
# **************************************************

# *** Question 3.4 ***
# Trading table
df_PT_insample1 = fn.tab_PT_insample(df_data=df_data, A='ZW Adj Close', B='ZC Adj Close', W=1000, L=2, in_level=1.5, stop_level=None)

# Plot metrics
fn.plot_PT_wealth_positions_leverage(df_PT=df_PT_insample1, question='3.4', method='IS', L=2)

# *** Question 3.5 ***
# Trading table
df_PT_insample2 = fn.tab_PT_insample(df_data=df_data, A='ZW Adj Close', B='ZC Adj Close', W=1000, L=20, in_level=1.5, stop_level=None)

# Plot metrics
fn.plot_PT_wealth_positions_leverage(df_PT=df_PT_insample2, question='3.5', method='IS', L=20)


# **************************************************
# *** QUESTION 3.2.2: Stop Loss                  ***
# **************************************************

# *** Question 3.6 ***
# Fit AR(1) model with const (drift)
model = sm.tsa.AutoReg(df_PT_insample1['Spread'], lags=1, trend='c').fit()

# Conditional distribution under stationarity
mu = model.params[0] / (1 - model.params[1])
var = model.sigma2 / ((1 - model.params[1]) ** 2)

# Report
print('\nProb(z_t+1 > z_stop=1.75) = {:.2%}'.format(1 - norm.cdf(x=1.75, loc=mu, scale=np.sqrt(var))))
print('Prob(z_t+1 > z_stop=2.75) = {:.2%}'.format(1 - norm.cdf(x=2.75, loc=mu, scale=np.sqrt(var))))

# *** Question 3.7 ***
df_PT_insample3 = fn.tab_PT_insample(df_data=df_data, A='ZW Adj Close', B='ZC Adj Close', W=1000, L=2, in_level=1.5, stop_level=2.75)

# Plot metrics
fn.plot_PT_wealth_positions_leverage(df_PT=df_PT_insample3, question='3.7', method='IS', L=2, zstop=2.75)


# **************************************************
# *** QUESTION 3.3: Out-of-Sample Pair Trading   ***
# **************************************************

# *** Question 3.8/9/10 ***

# Trading table
df_PT_outsample1 = fn.tab_PT_outsample(df_data=df_data, A='ZW Adj Close', B='ZC Adj Close', df_ts_coint=df_ts_coint_500, W=1000, L=2, in_level=1.5, stop_level=2.75, sample_size=500, pred_size=20, sig_coint=None)

# Plot metrics
fn.plot_PT_wealth_positions_leverage(df_PT=df_PT_outsample1, question='3.10', method='OS', L=2, zstop=2.75)

# Plot correlations
sns.set(context='paper', style='ticks', font_scale=1.0)
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot()
ax.grid(False)
ax.set_title(label='Rolling Correlations (OS)', size=28)
# Items
sns.lineplot(x=pd.to_datetime(df_PT_outsample1.index), y=df_PT_outsample1['Corr Prices'], label='Corr Prices', color='green', lw=3)
sns.lineplot(x=pd.to_datetime(df_PT_outsample1.index), y=df_PT_outsample1['Corr Returns'], label='Corr Log-Returns', color='firebrick', lw=3)
# X-axis settings
date_locator = mdates.YearLocator()
date_formatter = mdates.DateFormatter('%Y')
ax.tick_params(axis='x', labelrotation=0, labelsize=18)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(date_formatter)
ax.set_xlabel(xlabel='')
# Y-axis settings
ax.set_yticklabels(labels=['{:.1f}'.format(y) for y in ax.get_yticks()], size=18)
ax.set_ylabel(ylabel='')
# Legend settings
ax.legend(loc='lower left', fontsize=16)
# Show and save
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q3.9_rolling_correlations_os.png'))
plt.close()

# Plot alphas
sns.set(context='paper', style='ticks', font_scale=1.0)
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot()
ax.grid(False)
ax.set_title(label='Alpha OS vs. IS', size=28)
# Items
sns.lineplot(x=pd.to_datetime(df_PT_outsample1.index), y=df_PT_outsample1['Alpha'], label='Alpha OS', color='orange', lw=3)
sns.lineplot(x=pd.to_datetime(df_PT_insample3.index), y=df_PT_insample3['Alpha'], label='Alpha IS', color='red', lw=3)
# X-axis settings
date_locator = mdates.YearLocator()
date_formatter = mdates.DateFormatter('%Y')
ax.tick_params(axis='x', labelrotation=0, labelsize=18)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(date_formatter)
ax.set_xlabel(xlabel='')
# Y-axis settings
ax.set_yticklabels(labels=['{:.0f}'.format(y) for y in ax.get_yticks()], size=18)
ax.set_ylabel(ylabel='')
# Legend settings
ax.legend(loc='upper left', fontsize=16)
# Show and save
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q3.9_alphas_os.png'))
plt.close()

# Plot betas
sns.set(context='paper', style='ticks', font_scale=1.0)
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot()
ax.grid(False)
ax.set_title(label='Beta OS vs. IS', size=28)
# Items
sns.lineplot(x=pd.to_datetime(df_PT_outsample1.index), y=df_PT_outsample1['Beta'], label='Beta OS', color='orange', lw=3)
sns.lineplot(x=pd.to_datetime(df_PT_insample3.index), y=df_PT_insample3['Beta'], label='Beta IS', color='red', lw=3)
# X-axis settings
date_locator = mdates.YearLocator()
date_formatter = mdates.DateFormatter('%Y')
ax.tick_params(axis='x', labelrotation=0, labelsize=18)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(date_formatter)
ax.set_xlabel(xlabel='')
# Y-axis settings
ax.set_yticklabels(labels=['{:.1f}'.format(y) for y in ax.get_yticks()], size=18)
ax.set_ylabel(ylabel='')
# Legend settings
ax.legend(loc='upper left', fontsize=16)
# Show and save
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q3.9_betas_os.png'))
plt.close()

# Plot spreads
sns.set(context='paper', style='ticks', font_scale=1.0)
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot()
ax.grid(False)
ax.set_title(label='Spread OS vs. IS', size=28)
# Items
ax.axhline(y=0, color='black', ls='--', lw=2)
ax.axhline(y=1.5, color='green', ls='--', lw=2)
ax.axhline(y=-1.5, color='green', ls='--', lw=2)
ax.axhline(y=2.75, color='red', ls='--', lw=2)
ax.axhline(y=-2.75, color='red', ls='--', lw=2)
sns.lineplot(x=pd.to_datetime(df_PT_outsample1.index), y=df_PT_outsample1['Spread'], label='Spread OS', color='orange', lw=3)
sns.lineplot(x=pd.to_datetime(df_PT_insample3.index), y=df_PT_insample3['Spread'], label='Spread IS', color='lightcoral', lw=3)
# X-axis settings
date_locator = mdates.YearLocator()
date_formatter = mdates.DateFormatter('%Y')
ax.tick_params(axis='x', labelrotation=0, labelsize=18)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(date_formatter)
ax.set_xlabel(xlabel='')
# Y-axis settings
ax.set_yticklabels(labels=['{:.1f}'.format(y) for y in ax.get_yticks()], size=18)
ax.set_ylabel(ylabel='')
# Legend settings
ax.legend(loc='upper left', fontsize=16)
# Show and save
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q3.9_spreads_os.png'))
plt.close()




# *** Question 3.11/12 ***
# Trading table
df_PT_outsample2 = fn.tab_PT_outsample(df_data=df_data, A='ZW Adj Close', B='ZC Adj Close', df_ts_coint=df_ts_coint_500, W=1000, L=2, in_level=1.5, stop_level=2.75, sample_size=500, pred_size=20, sig_coint=0.1)

# Plot metrics
fn.plot_PT_wealth_positions_leverage(df_PT=df_PT_outsample2, question='3.12', method='OS', L=2, zstop=2.75, coint='10%')

# Plot p-values (stem plot)
fig = plt.figure(figsize=(12, 8), dpi=300)
ax = fig.add_subplot()
ax.set_title(label='Cointegration P-values (OS)', size=28)
# Items
plt.stem(pd.to_datetime(df_PT_insample2.index), df_PT_outsample2['Coint PV'], bottom=0.1)
# X-axis settings
date_locator = mdates.YearLocator()
date_formatter = mdates.DateFormatter('%Y')
ax.tick_params(axis='x', labelrotation=0, labelsize=18)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(date_formatter)
ax.set_xlabel(xlabel='')
# Y-axis settings
ax.set_yticklabels(labels=['{:.1f}'.format(y) for y in ax.get_yticks()], size=18)
ax.set_ylabel(ylabel='')
# Show and save
fig.tight_layout()
plt.show()
fig.savefig(Path.joinpath(paths.get('output'), 'Q3.11_cointegration_pv_os.png'))
plt.close()

# Table pair trading reports
df_PT_reports = pd.concat([fn.tab_PT_report(df_PT=df_PT_insample1, strategy='IS_L=2_zin=1.5'),
                           fn.tab_PT_report(df_PT=df_PT_insample2, strategy='IS_L=20_zin=1.5'),
                           fn.tab_PT_report(df_PT=df_PT_insample3, strategy='IS_L=2_zin=1.5_zstop=2.75'),
                           fn.tab_PT_report(df_PT=df_PT_outsample1, strategy='OS_L=2_zin=1.5_zstop=2.75'),
                           fn.tab_PT_report(df_PT=df_PT_outsample2, strategy='OS_L=2_zin=1.5_zstop=2.75_coint=10%')], axis=1)
df_PT_reports.to_latex(Path.joinpath(paths.get('output'), 'Q3_PT_reports.tex'))
