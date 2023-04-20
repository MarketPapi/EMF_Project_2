# Import packages
from itertools import permutations
from scipy import stats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm


# %%
# **************************************************
# *** Branch: Artur                              ***
# **************************************************

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
    # Rename the columns
    df_data.columns = ['ZC High', 'ZC Low', 'ZC Open', 'ZC Close', 'ZC Vol', 'ZC Adj Close',
                       'ZW High', 'ZW Low', 'ZW Open', 'ZW Close', 'ZW Vol', 'ZW Adj Close',
                       'ZS High', 'ZS Low', 'ZS Open', 'ZS Close', 'ZS Vol', 'ZS Adj Close',
                       'KC High', 'KC Low', 'KC Open', 'KC Close', 'KC Vol', 'KC Adj Close',
                       'CC High', 'CC Low', 'CC Open', 'CC Close', 'CC Vol', 'CC Adj Close']
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

        # White noise array.
        p_t = pd.Series(white_noise_agg[:, i])[1:]
        # Lagged White noise array.
        p_t_1 = pd.Series(white_noise_agg[:, i]).shift(1)[1:]
        T_1 = len(p_t)
        # Phi hat calculation
        phi_hat = p_t.cov(p_t_1) / p_t_1.var()
        # Standard error calculation
        u = p_t.mean() - phi_hat * p_t_1.mean()
        s2 = (1/(T_1-1))*sum((p_t - u - phi_hat*p_t_1)**2)
        phi_std = (s2 / sum((p_t_1 - p_t_1.mean()) ** 2)) ** 0.5

        # Step 4: Compute the T-Statistic
        df_stat = (phi_hat - 1) / phi_std

        # Save to DataFrame
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


def cointegration(df, column_names, permut=True):
    """
    Running a regression of log-prices on combinations of assets
    or a single pair of assets.

    :param df: DataFrame with prices.
    :return: DataFrame with alpha, beta and DF Test-Statistic.
    """

    # Create permutations with asset pairs, in a list [(0,1),(0,2)..]
    if permut:
        combos = list(permutations(range(len(list(df.columns))), 2))
    else:
        # Reg asset1=X on asset0=y
        combos = [(0, 1)]

    # Output DataFrame
    comm_coint = pd.DataFrame(index=['Alpha', 'Beta', 'DF_TS'])

    for i, j in combos:
        # Create asset pair name
        comm_name = str(column_names[i]) + '-' + str(column_names[j])

        # Regression between contemporaneous log-prices (i on j).
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

        # 2nd Regression (first difference on lagged residual).
        model_2 = sm.OLS(resid_fd, sm.add_constant(resid_lagged)).fit()

        # Output DF_Test
        phi_minus_1 = model_2.params[0]
        phi_std = model_2.bse[0]
        tstat = phi_minus_1 / phi_std

        # Arrange into DataFrame
        comm_coint[comm_name] = [alpha, beta, tstat]

    return comm_coint.T


def simulate_coint_cv(T, N):
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

        # Compute first difference
        resid_diff = pd.Series(resid - resid_shifted)
        resid_diff.dropna(inplace=True)

        # Regress first difference on lagged residual
        model_2 = sm.OLS(resid_diff, sm.add_constant(resid_shifted)).fit()

        # Compute DF_Tstat
        tstat_calc = model_2.params[0] / model_2.bse[0]

        # Append to list
        tstat.append(tstat_calc)

    return tstat


# %%
# **************************************************
# *** Branch: Anis                               ***
# **************************************************

def tab_autocorrelogram(s_data, alpha=0.05, max_lags=10):
    # Assumption: i.i.d. process ==> sqrt(T)*rho_k follows N(0,1) ==> rho_k follows N(0,1/T)
    T = len(s_data)
    std_rho_k = np.sqrt(1 / T)
    df_autocorrelogram = pd.DataFrame(columns=['Autocorrelation', 'CI @{:.0%}'.format(1 - alpha)])
    df_autocorrelogram.index.rename('Lags', inplace=True)
    for k in range(1, max_lags + 1):
        rho_k = s_data.autocorr(lag=k)
        crit_val = stats.norm.ppf((1 - alpha / 2), loc=0, scale=std_rho_k)
        conf_int = [-crit_val, crit_val]
        df_autocorrelogram.loc[k, 'Autocorrelation'] = rho_k
        df_autocorrelogram.loc[k, 'CI @{:.0%}'.format(1 - alpha)] = conf_int
    return df_autocorrelogram


def Ljung_Box_test(s_data, p=10):
    # Null: rho_1 = ... = rho_p = 0
    # Alternative: rho_k != 0 for some k = 1,...,p
    Q_p = 0
    T = len(s_data)
    for k in range(1, p + 1):
        rho_k = s_data.autocorr(lag=k)
        Q_p += (1 / (T - k)) * (rho_k ** 2)
    Q_p = (T * (T + 2)) * Q_p
    p_value = 1 - stats.chi2.cdf(Q_p, p)
    print('\nLjung-Box Test Report')
    print('Test stat (Q_p):', Q_p.round(2))
    print('P-value:', p_value.round(2))


def do_PT_accounting(df_PT, W, L):
    # Initialization
    df_PT = df_PT.copy()
    curr = df_PT.index[0]
    df_PT.loc[curr, 'SizeA'] = 0
    df_PT.loc[curr, 'SizeB'] = 0
    df_PT.loc[curr, 'Cash'] = W
    df_PT.loc[curr, 'Margin Account'] = 0
    df_PT.loc[curr, 'Long Securities'] = 0
    df_PT.loc[curr, 'Short Securities'] = 0
    df_PT.loc[curr, 'Equity'] = W
    ls_cols = ['SizeA', 'SizeB', 'Cash', 'Margin Account', 'Long Securities', 'Short Securities', 'Equity', 'Total Assets', 'Total Liabilities']

    # Accounting
    for i in range(len(df_PT.index)):
        # Bring forward balances
        curr = df_PT.index[i]
        if i > 0:
            prev = df_PT.index[i-1]
            for col in ls_cols:
                df_PT.loc[curr, col] = df_PT.loc[prev, col]

        # Position1
        if df_PT.loc[curr, 'Pos1 Active']:
            # Open position
            if df_PT.loc[curr, 'Pos1 Open']:
                # Deposit on margin account
                df_PT.loc[curr, 'Cash'] -= df_PT.loc[curr, 'Equity'] * (1 / (1 + L))
                df_PT.loc[curr, 'Margin Account'] += df_PT.loc[curr, 'Equity'] * (1 / (1 + L))
                # Short leg (A)
                df_PT.loc[curr, 'Margin Account'] += df_PT.loc[curr, 'Equity'] * (L / (1 + L))
                df_PT.loc[curr, 'Short Securities'] += df_PT.loc[curr, 'Equity'] * (L / (1 + L))
                df_PT.loc[curr, 'SizeA'] = df_PT.loc[curr, 'Short Securities'] / df_PT.loc[curr, 'PriceA']
                # Long leg (B)
                df_PT.loc[curr, 'Margin Account'] -= df_PT.loc[curr, 'Equity'] * (L / (1 + L))
                df_PT.loc[curr, 'Long Securities'] += df_PT.loc[curr, 'Equity'] * (L / (1 + L))
                df_PT.loc[curr, 'SizeB'] = df_PT.loc[curr, 'Long Securities'] / df_PT.loc[curr, 'PriceB']

            # Market-to-market position
            else:
                # Short leg (A)
                prev = df_PT.index[i-1]
                df_PT.loc[curr, 'Short Securities'] += (df_PT.loc[curr, 'PriceA'] - df_PT.loc[prev, 'PriceA']) * df_PT.loc[curr, 'SizeA']
                df_PT.loc[curr, 'Equity'] -= (df_PT.loc[curr, 'PriceA'] - df_PT.loc[prev, 'PriceA']) * df_PT.loc[curr, 'SizeA']
                # Long leg (B)
                df_PT.loc[curr, 'Long Securities'] += (df_PT.loc[curr, 'PriceB'] - df_PT.loc[prev, 'PriceB']) * df_PT.loc[curr, 'SizeB']
                df_PT.loc[curr, 'Equity'] += (df_PT.loc[curr, 'PriceB'] - df_PT.loc[prev, 'PriceB']) * df_PT.loc[curr, 'SizeB']

            # Close position (after MTM)
            if df_PT.loc[curr, 'Pos1 Close']:
                # Long leg (B)
                df_PT.loc[curr, 'Margin Account'] += df_PT.loc[curr, 'Long Securities']
                df_PT.loc[curr, 'Long Securities'] -= df_PT.loc[curr, 'Long Securities'] # Zeroed
                df_PT.loc[curr, 'SizeB'] = 0
                # Short leg (A)
                df_PT.loc[curr, 'Margin Account'] -= df_PT.loc[curr, 'Short Securities']
                df_PT.loc[curr, 'Short Securities'] -= df_PT.loc[curr, 'Short Securities'] # Zeroed
                df_PT.loc[curr, 'SizeA'] = 0
                # Withdraw from margin account
                df_PT.loc[curr, 'Cash'] += df_PT.loc[curr, 'Margin Account']
                df_PT.loc[curr, 'Margin Account'] -= df_PT.loc[curr, 'Margin Account'] # Zeroed

        # Position2
        elif df_PT.loc[curr, 'Pos2 Active']:
            # Open position
            if df_PT.loc[curr, 'Pos2 Open']:
                # Deposit on margin account
                df_PT.loc[curr, 'Cash'] -= df_PT.loc[curr, 'Equity'] * (1 / (1 + L))
                df_PT.loc[curr, 'Margin Account'] += df_PT.loc[curr, 'Equity'] * (1 / (1 + L))
                # Short leg (B)
                df_PT.loc[curr, 'Margin Account'] += df_PT.loc[curr, 'Equity'] * (L / (1 + L))
                df_PT.loc[curr, 'Short Securities'] += df_PT.loc[curr, 'Equity'] * (L / (1 + L))
                df_PT.loc[curr, 'SizeB'] = df_PT.loc[curr, 'Short Securities'] / df_PT.loc[curr, 'PriceB']
                # Long leg (A)
                df_PT.loc[curr, 'Margin Account'] -= df_PT.loc[curr, 'Equity'] * (L / (1 + L))
                df_PT.loc[curr, 'Long Securities'] += df_PT.loc[curr, 'Equity'] * (L / (1 + L))
                df_PT.loc[curr, 'SizeA'] = df_PT.loc[curr, 'Long Securities'] / df_PT.loc[curr, 'PriceA']

            # Market-to-market position
            else:
                # Short leg (B)
                prev = df_PT.index[i-1]
                df_PT.loc[curr, 'Short Securities'] += (df_PT.loc[curr, 'PriceB'] - df_PT.loc[prev, 'PriceB']) * df_PT.loc[curr, 'SizeB']
                df_PT.loc[curr, 'Equity'] -= (df_PT.loc[curr, 'PriceB'] - df_PT.loc[prev, 'PriceB']) * df_PT.loc[curr, 'SizeB']
                # Long leg (A)
                df_PT.loc[curr, 'Long Securities'] += (df_PT.loc[curr, 'PriceA'] - df_PT.loc[prev, 'PriceA']) * df_PT.loc[curr, 'SizeA']
                df_PT.loc[curr, 'Equity'] += (df_PT.loc[curr, 'PriceA'] - df_PT.loc[prev, 'PriceA']) * df_PT.loc[curr, 'SizeA']

            # Close position (after MTM)
            if df_PT.loc[curr, 'Pos2 Close']:
                # Long leg (A)
                df_PT.loc[curr, 'Margin Account'] += df_PT.loc[curr, 'Long Securities']
                df_PT.loc[curr, 'Long Securities'] -= df_PT.loc[curr, 'Long Securities'] # Zeroed
                df_PT.loc[curr, 'SizeA'] = 0
                # Short leg (B)
                df_PT.loc[curr, 'Margin Account'] -= df_PT.loc[curr, 'Short Securities']
                df_PT.loc[curr, 'Short Securities'] -= df_PT.loc[curr, 'Short Securities'] # Zeroed
                df_PT.loc[curr, 'SizeB'] = 0
                # Withdraw from margin account
                df_PT.loc[curr, 'Cash'] += df_PT.loc[curr, 'Margin Account']
                df_PT.loc[curr, 'Margin Account'] -= df_PT.loc[curr, 'Margin Account'] # Zeroed

        # Total assets and liabilities
        df_PT.loc[curr, 'Total Assets'] = np.sum([df_PT.loc[curr, col] for col in ['Cash', 'Margin Account', 'Long Securities']])
        df_PT.loc[curr, 'Total Liabilities'] = np.sum([df_PT.loc[curr, col] for col in ['Short Securities', 'Equity']])

    # Rounding
    for col in ls_cols:
        df_PT[col] = df_PT[col].round(2)

    # Return output
    return df_PT


def tab_PT_insample(df_data, A, B, W=1000, L=2, in_level=1.5, stop_level=None):
    # Initialization
    df_PT_insample = pd.DataFrame()
    df_PT_insample['PriceA'] = df_data[A]
    df_PT_insample['PriceB'] = df_data[B]

    # Spread
    X = df_PT_insample[['PriceB']]
    y = df_PT_insample['PriceA']
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    df_PT_insample['Alpha'] = lr_model.intercept_
    df_PT_insample['Beta'] = lr_model.coef_[0]
    df_PT_insample['Spread'] = y - lr_model.predict(X)
    # Normalization ==> we refer to normalized spread as spread
    df_PT_insample['Spread'] = df_PT_insample['Spread'] / df_PT_insample['Spread'].std(ddof=0)

    # Signals
    df_PT_insample['Sig1 Open'] = (df_PT_insample['Spread'] > in_level)
    df_PT_insample['Sig1 Close'] = (df_PT_insample['Spread'] <= 0)
    if stop_level is not None:
        df_PT_insample['Sig1 Stop'] = (df_PT_insample['Spread'] > stop_level)

    df_PT_insample['Sig2 Open'] = (df_PT_insample['Spread'] < -in_level)
    df_PT_insample['Sig2 Close'] = (df_PT_insample['Spread'] >= 0)
    if stop_level is not None:
        df_PT_insample['Sig2 Stop'] = (df_PT_insample['Spread'] < -stop_level)

    # Positions @t=0
    curr = df_PT_insample.index[0]
    if stop_level is None:
        # Position1
        df_PT_insample.loc[curr, ['Pos1 Active', 'Pos1 Open']] = df_PT_insample.loc[curr, 'Sig1 Open']
        df_PT_insample.loc[curr, 'Pos1 Close'] = False
        # Position2
        df_PT_insample.loc[curr, ['Pos2 Active', 'Pos2 Open']] = df_PT_insample.loc[curr, 'Sig2 Open']
        df_PT_insample.loc[curr, 'Pos2 Close'] = False

    else:
        # Position1
        df_PT_insample.loc[curr, ['Pos1 Active', 'Pos1 Open']] = (df_PT_insample.loc[curr, 'Sig1 Open'] and not df_PT_insample.loc[curr, 'Sig1 Stop'])
        df_PT_insample.loc[curr, 'Pos1 Close'] = False
        # Position2
        df_PT_insample.loc[curr, ['Pos2 Active', 'Pos2 Open']] = (df_PT_insample.loc[curr, 'Sig2 Open'] and not df_PT_insample.loc[curr, 'Sig2 Stop'])
        df_PT_insample.loc[curr, 'Pos2 Close'] = False

    # Positions @t>=1
    for i in range(1, len(df_PT_insample.index)):
        prev = df_PT_insample.index[i-1]
        curr = df_PT_insample.index[i]
        if stop_level is None:
            # Position1
            df_PT_insample.loc[curr, 'Pos1 Active'] = (df_PT_insample.loc[curr, 'Sig1 Open'] or (df_PT_insample.loc[prev, 'Pos1 Active'] and not df_PT_insample.loc[prev, 'Pos1 Close']))
            df_PT_insample.loc[curr, 'Pos1 Open'] = ((df_PT_insample.loc[curr, 'Pos1 Active'] and not df_PT_insample.loc[prev, 'Pos1 Active']) or
                                                     (df_PT_insample.loc[curr, 'Pos1 Active'] and df_PT_insample.loc[prev, 'Pos1 Close']))
            df_PT_insample.loc[curr, 'Pos1 Close'] = (df_PT_insample.loc[curr, 'Pos1 Active'] and df_PT_insample.loc[curr, 'Sig1 Close'])
            # Position2
            df_PT_insample.loc[curr, 'Pos2 Active'] = (df_PT_insample.loc[curr, 'Sig2 Open'] or (df_PT_insample.loc[prev, 'Pos2 Active'] and not df_PT_insample.loc[prev, 'Pos2 Close']))
            df_PT_insample.loc[curr, 'Pos2 Open'] = ((df_PT_insample.loc[curr, 'Pos2 Active'] and not df_PT_insample.loc[prev, 'Pos2 Active']) or
                                                     (df_PT_insample.loc[curr, 'Pos2 Active'] and df_PT_insample.loc[prev, 'Pos2 Close']))
            df_PT_insample.loc[curr, 'Pos2 Close'] = (df_PT_insample.loc[curr, 'Pos2 Active'] and df_PT_insample.loc[curr, 'Sig2 Close'])

        else:
            # Position1
            df_PT_insample.loc[curr, 'Pos1 Active'] = ((df_PT_insample.loc[curr, 'Sig1 Open'] and not df_PT_insample.loc[curr, 'Sig1 Stop']) or
                                                       (df_PT_insample.loc[prev, 'Pos1 Active'] and not df_PT_insample.loc[prev, 'Pos1 Close']))
            df_PT_insample.loc[curr, 'Pos1 Open'] = ((df_PT_insample.loc[curr, 'Pos1 Active'] and not df_PT_insample.loc[prev, 'Pos1 Active']) or
                                                     (df_PT_insample.loc[curr, 'Pos1 Active'] and df_PT_insample.loc[prev, 'Pos1 Close']))
            df_PT_insample.loc[curr, 'Pos1 Close'] = (df_PT_insample.loc[curr, 'Pos1 Active'] and (df_PT_insample.loc[curr, 'Sig1 Close'] or df_PT_insample.loc[curr, 'Sig1 Stop']))
            # Position2
            df_PT_insample.loc[curr, 'Pos2 Active'] = ((df_PT_insample.loc[curr, 'Sig2 Open'] and not df_PT_insample.loc[curr, 'Sig2 Stop']) or
                                                       (df_PT_insample.loc[prev, 'Pos2 Active'] and not df_PT_insample.loc[prev, 'Pos2 Close']))
            df_PT_insample.loc[curr, 'Pos2 Open'] = ((df_PT_insample.loc[curr, 'Pos2 Active'] and not df_PT_insample.loc[prev, 'Pos2 Active']) or
                                                     (df_PT_insample.loc[curr, 'Pos2 Active'] and df_PT_insample.loc[prev, 'Pos2 Close']))
            df_PT_insample.loc[curr, 'Pos2 Close'] = (df_PT_insample.loc[curr, 'Pos2 Active'] and (df_PT_insample.loc[curr, 'Sig2 Close'] or df_PT_insample.loc[curr, 'Sig2 Stop']))

    # Close open positions @t=T
    curr = df_PT_insample.index[-1]
    if df_PT_insample.loc[curr, 'Pos1 Active']:
        df_PT_insample.loc[curr, 'Pos1 Close'] = True

    if df_PT_insample.loc[curr, 'Pos2 Active']:
        df_PT_insample.loc[curr, 'Pos2 Close'] = True

    # Format as numpy.bool
    for col in df_PT_insample.columns[df_PT_insample.columns.get_loc('Sig1 Open'):]:
        df_PT_insample[col] = df_PT_insample[col].astype('bool')

    # Accounting
    df_PT_insample = do_PT_accounting(df_PT=df_PT_insample, W=W, L=L)

    # Return output
    return df_PT_insample


def tab_PT_outsample(df_data, A, B, df_ts_coint, W=1000, L=2, in_level=1.5, stop_level=None, sample_size=500, pred_size=20, sig_coint=None):
    # Initialization
    df_PT_outsample = pd.DataFrame()
    df_PT_outsample['PriceA'] = df_data[A]
    df_PT_outsample['PriceB'] = df_data[B]
    # Methodology: we use log returns
    df_PT_outsample['ReturnA'] = np.log(df_PT_outsample['PriceA']) - np.log(df_PT_outsample['PriceA'].shift(1))
    df_PT_outsample['ReturnB'] = np.log(df_PT_outsample['PriceB']) - np.log(df_PT_outsample['PriceB'].shift(1))

    # Rolling windows
    sample_start = 0
    sample_stop = sample_start + sample_size
    pred_start = sample_stop
    pred_stop = pred_start + pred_size
    ls_windows = [(range(sample_start, sample_stop), range(pred_start, pred_stop))]

    while pred_stop < len(df_PT_outsample.index):
        sample_start += pred_size
        sample_stop += pred_size
        pred_start += pred_size
        pred_stop = pred_start
        while (pred_stop < len(df_PT_outsample.index)) and (len(range(pred_start, pred_stop)) < pred_size):
            pred_stop += 1
        ls_windows.append((range(sample_start, sample_stop), range(pred_start, pred_stop)))

    # Rolling correlations
    df_PT_outsample['Corr Prices'] = np.nan
    df_PT_outsample['Corr Returns'] = np.nan
    for window in ls_windows:
        df_sample = df_PT_outsample.iloc[window[0]]
        df_PT_outsample.iloc[window[1], df_PT_outsample.columns.get_loc('Corr Prices')] = df_sample['PriceA'].corr(df_sample['PriceB'])
        df_PT_outsample.iloc[window[1], df_PT_outsample.columns.get_loc('Corr Returns')] = df_sample['ReturnA'].corr(df_sample['ReturnB'])

    # Spread
    df_PT_outsample['Alpha'] = np.nan
    df_PT_outsample['Beta'] = np.nan
    df_PT_outsample['Spread'] = np.nan
    for window in ls_windows:
        # Fit on sample
        df_sample = df_PT_outsample.iloc[window[0]]
        X = df_sample[['PriceB']]
        y = df_sample['PriceA']
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        df_sample['Spread'] = y - lr_model.predict(X)
        # Extend on pred
        df_pred = df_PT_outsample.iloc[window[1]]
        X = df_pred[['PriceB']]
        y = df_pred['PriceA']
        df_PT_outsample.iloc[window[1], df_PT_outsample.columns.get_loc('Alpha')] = lr_model.intercept_
        df_PT_outsample.iloc[window[1], df_PT_outsample.columns.get_loc('Beta')] = lr_model.coef_[0]
        # Normalization ==> use standard deviation estimated on estimation period (sample)
        df_PT_outsample.iloc[window[1], df_PT_outsample.columns.get_loc('Spread')] = (y - lr_model.predict(X)) / df_sample['Spread'].std(ddof=0)

    # Test cointegration
    if sig_coint is not None:
        l_adj_close_price = ['ZC Adj Close', 'ZW Adj Close', 'ZS Adj Close', 'KC Adj Close', 'CC Adj Close']
        df_data_ln = log_transform_cols(df=df_data, cols=l_adj_close_price)
        df_PT_outsample['Coint TS'] = np.nan
        df_PT_outsample['Coint PV'] = np.nan
        for window in ls_windows:
            df_sample = df_data_ln.iloc[window[0]]
            df_coint = cointegration(df=df_sample[[A, B]], column_names=[A, B], permut=False)
            tstat = df_coint.iloc[0, df_coint.columns.get_loc('DF_TS')]
            df_PT_outsample.iloc[window[1], df_PT_outsample.columns.get_loc('Coint TS')] = tstat
            df_PT_outsample.iloc[window[1], df_PT_outsample.columns.get_loc('Coint PV')] = get_pvalue(distribution=df_ts_coint, value=tstat)

    # Signals
    df_PT_outsample['Sig1 Open'] = (df_PT_outsample['Spread'] > in_level)
    df_PT_outsample['Sig1 Close'] = (df_PT_outsample['Spread'] <= 0)
    if stop_level is not None:
        df_PT_outsample['Sig1 Stop'] = (df_PT_outsample['Spread'] > stop_level)

    df_PT_outsample['Sig2 Open'] = (df_PT_outsample['Spread'] < -in_level)
    df_PT_outsample['Sig2 Close'] = (df_PT_outsample['Spread'] >= 0)
    if stop_level is not None:
        df_PT_outsample['Sig2 Stop'] = (df_PT_outsample['Spread'] < -stop_level)

    if sig_coint is not None:
        df_PT_outsample['Coint @{:.0%}'.format(sig_coint)] = (df_PT_outsample['Coint PV'] < sig_coint)

    # Positions @t=0
    curr = df_PT_outsample.index[0]
    if stop_level is None:
        if sig_coint is None:
            # Position1
            df_PT_outsample.loc[curr, ['Pos1 Active', 'Pos1 Open']] = df_PT_outsample.loc[curr, 'Sig1 Open']
            df_PT_outsample.loc[curr, 'Pos1 Close'] = False
            # Position2
            df_PT_outsample.loc[curr, ['Pos2 Active', 'Pos2 Open']] = df_PT_outsample.loc[curr, 'Sig2 Open']
            df_PT_outsample.loc[curr, 'Pos2 Close'] = False
        else:
            # Position1
            df_PT_outsample.loc[curr, ['Pos1 Active', 'Pos1 Open']] = (df_PT_outsample.loc[curr, 'Sig1 Open'] and df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)])
            df_PT_outsample.loc[curr, 'Pos1 Close'] = False
            # Position2
            df_PT_outsample.loc[curr, ['Pos2 Active', 'Pos2 Open']] = (df_PT_outsample.loc[curr, 'Sig2 Open'] and df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)])
            df_PT_outsample.loc[curr, 'Pos2 Close'] = False

    else:
        if sig_coint is None:
            # Position1
            df_PT_outsample.loc[curr, ['Pos1 Active', 'Pos1 Open']] = (df_PT_outsample.loc[curr, 'Sig1 Open'] and not df_PT_outsample.loc[curr, 'Sig1 Stop'])
            df_PT_outsample.loc[curr, 'Pos1 Close'] = False
            # Position2
            df_PT_outsample.loc[curr, ['Pos2 Active', 'Pos2 Open']] = (df_PT_outsample.loc[curr, 'Sig2 Open'] and not df_PT_outsample.loc[curr, 'Sig2 Stop'])
            df_PT_outsample.loc[curr, 'Pos2 Close'] = False
        else:
            # Position1
            df_PT_outsample.loc[curr, ['Pos1 Active', 'Pos1 Open']] = ((df_PT_outsample.loc[curr, 'Sig1 Open'] and not df_PT_outsample.loc[curr, 'Sig1 Stop']) and df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)])
            df_PT_outsample.loc[curr, 'Pos1 Close'] = False
            # Position2
            df_PT_outsample.loc[curr, ['Pos2 Active', 'Pos2 Open']] = ((df_PT_outsample.loc[curr, 'Sig2 Open'] and not df_PT_outsample.loc[curr, 'Sig2 Stop']) and df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)])
            df_PT_outsample.loc[curr, 'Pos2 Close'] = False

    # Positions @t>=1
    for i in range(1, len(df_PT_outsample.index)):
        prev = df_PT_outsample.index[i-1]
        curr = df_PT_outsample.index[i]
        if stop_level is None:
            if sig_coint is None:
                # Position1
                df_PT_outsample.loc[curr, 'Pos1 Active'] = (df_PT_outsample.loc[curr, 'Sig1 Open'] or (df_PT_outsample.loc[prev, 'Pos1 Active'] and not df_PT_outsample.loc[prev, 'Pos1 Close']))
                df_PT_outsample.loc[curr, 'Pos1 Open'] = ((df_PT_outsample.loc[curr, 'Pos1 Active'] and not df_PT_outsample.loc[prev, 'Pos1 Active']) or
                                                          (df_PT_outsample.loc[curr, 'Pos1 Active'] and df_PT_outsample.loc[prev, 'Pos1 Close']))
                df_PT_outsample.loc[curr, 'Pos1 Close'] = (df_PT_outsample.loc[curr, 'Pos1 Active'] and df_PT_outsample.loc[curr, 'Sig1 Close'])
                # Position2
                df_PT_outsample.loc[curr, 'Pos2 Active'] = (df_PT_outsample.loc[curr, 'Sig2 Open'] or (df_PT_outsample.loc[prev, 'Pos2 Active'] and not df_PT_outsample.loc[prev, 'Pos2 Close']))
                df_PT_outsample.loc[curr, 'Pos2 Open'] = ((df_PT_outsample.loc[curr, 'Pos2 Active'] and not df_PT_outsample.loc[prev, 'Pos2 Active']) or
                                                          (df_PT_outsample.loc[curr, 'Pos2 Active'] and df_PT_outsample.loc[prev, 'Pos2 Close']))
                df_PT_outsample.loc[curr, 'Pos2 Close'] = (df_PT_outsample.loc[curr, 'Pos2 Active'] and df_PT_outsample.loc[curr, 'Sig2 Close'])
            else:
                # Position1
                df_PT_outsample.loc[curr, 'Pos1 Active'] = ((df_PT_outsample.loc[curr, 'Sig1 Open'] and df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)]) or
                                                            (df_PT_outsample.loc[prev, 'Pos1 Active'] and not df_PT_outsample.loc[prev, 'Pos1 Close']))
                df_PT_outsample.loc[curr, 'Pos1 Open'] = ((df_PT_outsample.loc[curr, 'Pos1 Active'] and not df_PT_outsample.loc[prev, 'Pos1 Active']) or
                                                          (df_PT_outsample.loc[curr, 'Pos1 Active'] and df_PT_outsample.loc[prev, 'Pos1 Close']))
                df_PT_outsample.loc[curr, 'Pos1 Close'] = (df_PT_outsample.loc[curr, 'Pos1 Active'] and (df_PT_outsample.loc[curr, 'Sig1 Close'] or not df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)]))
                # Position2
                df_PT_outsample.loc[curr, 'Pos2 Active'] = ((df_PT_outsample.loc[curr, 'Sig2 Open'] and df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)]) or
                                                            (df_PT_outsample.loc[prev, 'Pos2 Active'] and not df_PT_outsample.loc[prev, 'Pos2 Close']))
                df_PT_outsample.loc[curr, 'Pos2 Open'] = ((df_PT_outsample.loc[curr, 'Pos2 Active'] and not df_PT_outsample.loc[prev, 'Pos2 Active']) or
                                                          (df_PT_outsample.loc[curr, 'Pos2 Active'] and df_PT_outsample.loc[prev, 'Pos2 Close']))
                df_PT_outsample.loc[curr, 'Pos2 Close'] = (df_PT_outsample.loc[curr, 'Pos2 Active'] and (df_PT_outsample.loc[curr, 'Sig2 Close'] or not df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)]))

        else:
            if sig_coint is None:
                # Position1
                df_PT_outsample.loc[curr, 'Pos1 Active'] = ((df_PT_outsample.loc[curr, 'Sig1 Open'] and not df_PT_outsample.loc[curr, 'Sig1 Stop']) or
                                                            (df_PT_outsample.loc[prev, 'Pos1 Active'] and not df_PT_outsample.loc[prev, 'Pos1 Close']))
                df_PT_outsample.loc[curr, 'Pos1 Open'] = ((df_PT_outsample.loc[curr, 'Pos1 Active'] and not df_PT_outsample.loc[prev, 'Pos1 Active']) or
                                                          (df_PT_outsample.loc[curr, 'Pos1 Active'] and df_PT_outsample.loc[prev, 'Pos1 Close']))
                df_PT_outsample.loc[curr, 'Pos1 Close'] = (df_PT_outsample.loc[curr, 'Pos1 Active'] and (df_PT_outsample.loc[curr, 'Sig1 Close'] or df_PT_outsample.loc[curr, 'Sig1 Stop']))
                # Position2
                df_PT_outsample.loc[curr, 'Pos2 Active'] = ((df_PT_outsample.loc[curr, 'Sig2 Open'] and not df_PT_outsample.loc[curr, 'Sig2 Stop']) or
                                                            (df_PT_outsample.loc[prev, 'Pos2 Active'] and not df_PT_outsample.loc[prev, 'Pos2 Close']))
                df_PT_outsample.loc[curr, 'Pos2 Open'] = ((df_PT_outsample.loc[curr, 'Pos2 Active'] and not df_PT_outsample.loc[prev, 'Pos2 Active']) or
                                                          (df_PT_outsample.loc[curr, 'Pos2 Active'] and df_PT_outsample.loc[prev, 'Pos2 Close']))
                df_PT_outsample.loc[curr, 'Pos2 Close'] = (df_PT_outsample.loc[curr, 'Pos2 Active'] and (df_PT_outsample.loc[curr, 'Sig2 Close'] or df_PT_outsample.loc[curr, 'Sig2 Stop']))
            else:
                # Position1
                df_PT_outsample.loc[curr, 'Pos1 Active'] = (((df_PT_outsample.loc[curr, 'Sig1 Open'] and df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)]) and not df_PT_outsample.loc[curr, 'Sig1 Stop']) or
                                                            (df_PT_outsample.loc[prev, 'Pos1 Active'] and not df_PT_outsample.loc[prev, 'Pos1 Close']))
                df_PT_outsample.loc[curr, 'Pos1 Open'] = ((df_PT_outsample.loc[curr, 'Pos1 Active'] and not df_PT_outsample.loc[prev, 'Pos1 Active']) or
                                                          (df_PT_outsample.loc[curr, 'Pos1 Active'] and df_PT_outsample.loc[prev, 'Pos1 Close']))
                df_PT_outsample.loc[curr, 'Pos1 Close'] = (df_PT_outsample.loc[curr, 'Pos1 Active'] and (df_PT_outsample.loc[curr, 'Sig1 Close'] or df_PT_outsample.loc[curr, 'Sig1 Stop'] or not df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)]))
                # Position2
                df_PT_outsample.loc[curr, 'Pos2 Active'] = (((df_PT_outsample.loc[curr, 'Sig2 Open'] and df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)]) and not df_PT_outsample.loc[curr, 'Sig2 Stop']) or
                                                            (df_PT_outsample.loc[prev, 'Pos2 Active'] and not df_PT_outsample.loc[prev, 'Pos2 Close']))
                df_PT_outsample.loc[curr, 'Pos2 Open'] = ((df_PT_outsample.loc[curr, 'Pos2 Active'] and not df_PT_outsample.loc[prev, 'Pos2 Active']) or
                                                          (df_PT_outsample.loc[curr, 'Pos2 Active'] and df_PT_outsample.loc[prev, 'Pos2 Close']))
                df_PT_outsample.loc[curr, 'Pos2 Close'] = (df_PT_outsample.loc[curr, 'Pos2 Active'] and (df_PT_outsample.loc[curr, 'Sig2 Close'] or df_PT_outsample.loc[curr, 'Sig2 Stop'] or not df_PT_outsample.loc[curr, 'Coint @{:.0%}'.format(sig_coint)]))

    # Close open positions @t=T
    curr = df_PT_outsample.index[-1]
    if df_PT_outsample.loc[curr, 'Pos1 Active']:
        df_PT_outsample.loc[curr, 'Pos1 Close'] = True

    if df_PT_outsample.loc[curr, 'Pos2 Active']:
        df_PT_outsample.loc[curr, 'Pos2 Close'] = True

    # Format as numpy.bool
    for col in df_PT_outsample.columns[df_PT_outsample.columns.get_loc('Sig1 Open'):]:
        df_PT_outsample[col] = df_PT_outsample[col].astype('bool')

    # Accounting
    df_PT_outsample = do_PT_accounting(df_PT=df_PT_outsample, W=W, L=L)

    # Return output
    return df_PT_outsample


def tab_PT_report(df_PT, strategy):
    df_PT_report = pd.DataFrame(columns=strategy)
    df_PT_report.loc['Profit'] = ':.2f'.format(df_PT.iloc[-1, df_PT.columns.get_loc('Equity')] - df_PT.iloc[0, df_PT.columns.get_loc('Equity')])
    df_PT_report.loc['ROE'] = ':.2%'.format((df_PT.iloc[-1, df_PT.columns.get_loc('Equity')] - df_PT.iloc[0, df_PT.columns.get_loc('Equity')]) / df_PT.iloc[0, df_PT.columns.get_loc('Equity')])
    df_PT_report.loc['Init wealth'] = ':.2f'.format(df_PT.iloc[0, df_PT.columns.get_loc('Equity')])
    df_PT_report.loc['Final wealth'] = ':.2f'.format(df_PT.iloc[-1, df_PT.columns.get_loc('Equity')])
    df_PT_report.loc['Min wealth'] = ':.2f'.format(df_PT['Equity'].min())
    df_PT_report.loc['Max wealth'] = ':.2f'.format(df_PT['Equity'].max())
    df_PT_report.loc['Pos1 trades'] = ':.0f'.format(df_PT['Pos1 Open'].astype(int).sum())
    df_PT_report.loc['Pos2 trades'] = ':.0f'.format(df_PT['Pos2 Open'].astype(int).sum())
    df_PT_report.loc['Total trades'] = ':.0f'.format(df_PT['Pos1 Open'].astype(int).sum() + df_PT['Pos2 Open'].astype(int).sum())
