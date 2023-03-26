import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from decimal import Decimal
from scipy import stats
from pathlib import Path
from scripts.project_parameters import paths
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


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
    ar_parameters = pd.DataFrame(columns=['AR_Coeff', 'AR_Coeff_SD', 'Trend_Coeff', 'Trend_Coeff_SD', 'DF_TS'])

    # Select P(0)
    p0 = df[column][0]
    p = np.zeros((T, N))
    p[0] = p0

    for i in range(0, N):

        # Step 2: Compute Random Walk
        for t in range(1, T):
            p[t, i] = p[t - 1, i] + white_noise[t, i]

        # Step 3: Estimate AR(1) Model
        ar_model = AutoReg(p[:, i], lags=1, trend='c').fit()
        phi_hat = ar_model.params[1]
        phi_std = ar_model.bse[1]
        trend = ar_model.params[0]
        trend_std = ar_model.bse[0]

        # Step 4: Compute the T-Statistic
        df_stat = (phi_hat - 1) / phi_std

        ar_parameters.loc[i] = [phi_hat, phi_std, trend, trend_std, df_stat]
    # Computing the critical values

    test_statistics = ar_parameters['DF_TS'].quantile([0.01, 0.05, 0.1])
    test_statistics = test_statistics.rename('T-Statistic')

    return test_statistics, ar_parameters


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
