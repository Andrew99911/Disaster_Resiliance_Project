import pandas as pd
import numpy as np
import os
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.offline as py

def gen_analysis(states):
    """
    Args:
        states ([type]): [description]

    Returns:
        [type]: [description]
    """

    return None

d = {'K': 3, 'k': 3, 'M': 6, 'm': 6, 'B': 9, 'b': 9}
def _str_to_num(s):
    """ Helper function for formatting the property damage and crop damage columns in get_df. Given a string of the 
    format "1.234M" or something similar (i.e. ending with B, M, K), will return the integer equivalent.

    Args:
        s (int or string): If a string, presumably something that ends with K, M or B. If s is not a string, it will 
        return the value (assumed to be in integer in this case).

    Returns:
        float/int: numeric representation of the string provided.
    """

    if isinstance(s, str) and s[-1] in d: 
        # if its a string where the final character is K, M, or B, convert accordingly
        num, magnitude = s[:-1], s[-1]
        return float(num) * 10 ** d[magnitude]
    elif isinstance(s, str): 
        # if just a normal string (assumed to be numeric), return the float representation
        return float(s)
    else: # otherwise, we just return
        return s

def _get_df(states):
    """Queries data from the 'data' subdirectory where the state of the row is in the list of states provided as a 
    parameter. The produced dataframe is converted into a dataframe where the injuries, property damange, crop damage, 
    and deaths are saved as columns and indexed by the datetime of that storm.

    Args:
        states (List of str): List of states we want to keep in the dataframe.

    Return:
        pandas.DataFrame: Contains the injuries, deaths, property damage and crop damage of each storm for the states
        given. NOTE: The metrics are resampled to monthly sums.
    """

    # subdirectory containing data
    subdir = "data"
    
    # columns to keep from the spreadsheets
    columns = ['STATE', 'END_DATE_TIME', 'INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT', 
        'DAMAGE_PROPERTY', 'DAMAGE_CROPS',]

    # gets list of spreadsheets (NOTE: its assumed that data directory only contains relevant spreadsheets)
    sheets_paths = os.listdir(subdir)

    # make a df from all the spreadsheets in the data subdirectory, keeping only the columns specified
    df = pd.concat([pd.read_csv(os.path.join(subdir, f), usecols=columns) for f in sheets_paths],ignore_index=True)

    # NOTE: all code past this point just cleans up dataframe for analysis

    df = df[df['STATE'].isin(states)] # only keep data from states specified
    
    df.reset_index(drop=True, inplace=True) # removing the indices, will be replaced with datetime indices

    # set the end date of storm events as the new index
    df.set_index('END_DATE_TIME', inplace=True)
    df.index = pd.to_datetime(df.index) # convert to datetime objects
    df.sort_index(inplace=True) # sort df for viewing/debugging

    # convert damage columns to integer values using helper function - refer to function _str_to_num for explanation
    df['DAMAGE_CROPS'] = df['DAMAGE_CROPS'].map(_str_to_num)
    df['DAMAGE_PROPERTY'] = df['DAMAGE_PROPERTY'].map(_str_to_num)

    df.fillna(0, inplace=True) # replace empty values with 0

    # change to total injuries, deaths, damage (remove direct/indirect)
    df['INJURIES'] = df['INJURIES_DIRECT'] + df['INJURIES_INDIRECT']
    df['DEATHS'] = df['DEATHS_DIRECT'] + df['DEATHS_INDIRECT']
    df['DAMAGE'] = df['DAMAGE_PROPERTY'] + df['DAMAGE_CROPS']

    # return the df with unimportant columns droppped and resampled by month and summed (i.e. gives monthly sums instead
    # of individual data points)
    return df.drop(columns=['STATE', 'INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT']).resample('M').sum()

def _gen_fbprophet_model(df, use_outliers=True):
    """Using the Prophet package from Facebook, generate two interactive plotly graphs projecting damage for the next 
    56 months (i.e. until 2025). First graph is the explicit forecast, the second graph contains the trend and 
    seasonality components.

    Args:
        df (pandas.DataFrame): DataFrame containing monthly sums to analyze/forecast
        use_outliers(Boolean): whether or not to filter out outliers before analysis

    Return:
        plotly figures: two plotly figures for forecasting and components of forecast
    """

    series = df['DAMAGE'] # select the damage series to analyze

    if not use_outliers:
        series = series[np.abs(stats.zscore(series)) < 2] # keep values within two standard deviations

    # create a temporary dataframe for fbprophet to fit
    temp = pd.DataFrame()
    temp['ds'] = series.index
    temp['y'] = series.values

    m = Prophet()
    m.fit(temp)

    # create a future dataframe to predict the next 36 months
    future = m.make_future_dataframe(periods=36, freq='M')

    # make the prediction
    forecast = m.predict(future)

    # generate the two ploty figures, first showing forecast, second showing forecast components
    fig = plot_plotly(m, forecast)
    fig2 = plot_components_plotly(m, forecast)

    return fig, fig2

def _significant_months_barchart(df):

    return None

def _monthly_cost_hist(df):

    return None