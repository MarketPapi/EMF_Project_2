from decimal import Decimal
from itertools import permutations
from pathlib import Path
from scipy import stats
from scripts.project_parameters import paths
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


def read_data(file_path, skiprows=0):
    """
    Importing the data from csv file.

    :param file_path: Str, file path
    :param skiprows: Int, number of rows to skip
    """
    # Read the data file
    df_data = pd.read_csv(file_path, skiprows=skiprows)
    # Transform to datetime format
    df_data['Date'] = pd.to_datetime(df_data['Date'], format='%Y-%m-%d')
    # Hide timestamp formatting
    df_data['Date'] = df_data['Date'].dt.strftime('%Y-%m-%d')
    # Set new index
    df_data.set_index('Date', inplace=True)
    df_data.index.rename('DATE', inplace=True)

    # Rename the columns
    df_data.columns = ['ZC High', 'ZC Low', 'ZC Open', 'ZC Close', 'ZC Vol', 'ZC Adj Close',
                       'ZW High', 'ZW Low', 'ZW Open', 'ZW Close', 'ZW Vol', 'ZW Adj Close',
                       'ZS High', 'ZS Low', 'ZS Open', 'ZS Close', 'ZS Vol', 'ZS Adj Close',
                       'KC High', 'KC Low', 'KC Open', 'KC Close', 'KC Vol', 'KC Adj Close',
                       'CC High', 'CC Low', 'CC Open', 'CC Close', 'CC Vol', 'CC Adj Close',
                       ]
    return df_data


def log_transform_cols(df, cols):
    """
    Takes the log transformation of specified columns in a DataFrame.
    :param df: pandas DataFrame
    :param cols: list of column names to transform
    :return: new pandas DataFrame with transformed columns
    """
    # Create copy of the DataFrame
    df_new = df.copy()
    # Loop through list of columns and transform
    for col in cols:
        df_new[col] = np.log(df_new[col])
    return df_new


def reg(df, column, lag=0):
    """
    Returns the results of a regression on a lagged feature

    :param df: DataFrame
    :param column: Name of the column as a list
    :param lag: Integer representing lag
    :return: Test statistic of the regression
    """

    # New DataFrame
    new_df = pd.DataFrame(df[column].copy())

    # Creating lagged feature
    new_df['Lagged'] = new_df[column].shift(lag)
    new_df.dropna(inplace=True)

    X = new_df['Lagged'].values
    y = new_df[column].values

    # Run linear regression
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    reg_results = model.fit()

    # Computing the T-Stat from the regression parameters
    t_stat_data = (reg_results.params[1] - 1) / reg_results.bse[1]

    return t_stat_data


def critical_value(df, column, T, N):
    """
    Returns parameters of AR(1) form the Monte-Carlo Distribution
    Returns the critical values from the Monte-Carlo Distribution

    :param df: DataFrame from which to get p0.
    :param column: Column name of P0.
    :param T: Length of the timeseries to generate shocks.
    :param N: Number of simulations.
    :param white_noise: white noise array.
    :return: DataFrame for the AR(1) parameters and the test-statistics for the generated distribution.
    """

    # Generate white noise
    np.random.seed(23031997)
    white_noise = np.random.normal(0, 1, size=(T, N))

    # Output DataFrame
    ar_parameters = pd.DataFrame(columns=['AR_Coeff', 'AR_Coeff_SD', 'DF_TS'])

    # Select P(0)
    p0 = df[column][0]
    white_noise[0] = p0

    # Aggregate the shocks
    white_noise_agg = white_noise.cumsum(axis=0)

    for i in tqdm(range(0, N), desc="Simulating Test Statistics"):
        # Step 3: Estimate AR(1) Model
        ar_model = AutoReg(white_noise_agg[:, i], lags=1, trend='c').fit()
        phi_hat = ar_model.params[1]
        phi_std = ar_model.bse[1]

        # Step 4: Compute the T-Statistic
        df_stat = (phi_hat - 1) / phi_std

        ar_parameters.loc[i] = [phi_hat, phi_std, df_stat]

    # Computing the critical values
    critical_val = ar_parameters['DF_TS'].quantile([0.01, 0.05, 0.1])
    critical_val = critical_val.rename('Critical Value')

    return critical_val, ar_parameters


def get_pvalue(distribution, value):
    """
    Given the distribution of t-stats and the observed value it return the quantile
    to which the value corresponds.
    :param distribution: array of the distribution values
    :param value: Observed value
    :return: Quantile (p_value)
    """
    quantiles = np.percentile(distribution, np.arange(0, 100, 1))
    index = np.searchsorted(quantiles, value)
    quantile = (index / 100.0)
    return quantile


def format_float(df):
    """
    Format the floats to two decimal places in :param df: DataFrame
    :return: Formatted DataFrame
    """
    df_new = df.copy()
    df_new = df_new.applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, (int, float)) else x)
    return df_new


def cointgration(df):
    """
    Running a regression of log-prices on combinations of assets.

    :param df: DataFrame with log prices.
    :return: DataFrame with alpha, beta and DF Test-Statistic.
    """

    cols = list(df.columns)
    combos = list(permutations(range(len(cols)), 2))

    comm_names = ['Corn', 'Wheat', 'Soybean', 'Coffee', 'Cacao']
    comm_coint = pd.DataFrame(index=['Alpha', 'Beta', 'DF_TS'])

    for i, j in combos:
        comm_name = str(comm_names[i]) + '-' + str(comm_names[j])

        # Regression between contemporaneous log-prices.
        # Regressing i on j
        asset1 = df.iloc[:, i]
        asset2 = df.iloc[:, j]
        model_1 = sm.OLS(asset1, sm.add_constant(asset2)).fit()
        alpha = model_1.params['const']
        beta = model_1.params[1]

        # Residuals
        resid = pd.Series(model_1.resid)
        resid_lagged = resid.shift(1)
        resid_lagged.dropna(inplace=True)
        resid_fd = pd.Series(resid - resid_lagged)
        resid_fd.dropna(inplace=True)

        # 2nd Regression
        model_2 = sm.OLS(resid_fd, sm.add_constant(resid_lagged)).fit()

        # Output DF_Test
        phi_minus_1 = model_2.params[0]
        phi_std = model_2.bse[0]
        tstat = phi_minus_1 / phi_std

        # Arrange into DataFrame
        comm_coint[comm_name] = [alpha, beta, tstat]

    return comm_coint.T

def simulate_coint_cv(T,N):
    """
    Simulate test statistics.
    :param T: Time series length.
    :param N: Number of simulations
    :return: list of test statistics.
    """

    np.random.seed(23031998)
    white_noise_A = np.random.normal(0, 1, size=(T, N))
    white_noise_A[0] = 0
    white_noise_A_agg = white_noise_A.cumsum(axis=0)

    np.random.seed(23031999)
    white_noise_B = np.random.normal(0, 1, size=(T, N))
    white_noise_B[0] = 0
    white_noise_B_agg = white_noise_B.cumsum(axis=0)

    pA = white_noise_A_agg
    pB = white_noise_B_agg

    tstat = []

    for i in tqdm(range(0, N), desc="Simulating Test Statistics"):

        # Regress pA on pB
        model_1 = sm.OLS(pA[:, i], sm.add_constant(pB[:, i])).fit()

        # Extract residuals
        resid = pd.Series(model_1.resid)

        # Compute lagged residuals
        resid_shifted = resid.shift(1)
        resid_shifted.dropna(inplace=True)

        #Compute first difference
        resid_diff = pd.Series(resid - resid_shifted)
        resid_diff.dropna(inplace=True)

        # Regress first difference on lagged residual
        model_2 = sm.OLS(resid_diff, sm.add_constant(resid_shifted)).fit()

        # Compute DF_Tstat
        tstat_calc = model_2.params[0] / model_2.bse[0]

        #Append to list
        tstat.append(tstat_calc)

    return tstat

