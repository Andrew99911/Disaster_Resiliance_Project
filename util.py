import pandas as pd
import numpy as np
import os
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px
import calendar

def gen_analysis(states):
    """ Calls all the helper functions and returns the dictionary containing all the figures. Use 'fbprophet' to get a
    tuple of predictions and components, 'top10' to get a bar chart of top 10 historical months by damage, and
    'month_dist' to get a bar chart of distribution of damage by month.
    
    Args:
        states (list of str): List of states to analyze. Assumed to all be fully capitalized.

    Returns:
        dict of str to plotly graphs: contains all the relevant plotly graphs
    """

    ret = {} # dictionary to hold all the graphs
    df = _get_df(states)

    # use these 
    ret['fbprophet'] = _gen_fbprophet_model(df, use_outliers=True)
    ret['top10'] = _significant_months_barchart(df, metric='damage')
    ret['month_dist'] = _monthly_dist(df, metric='damage')

    return ret


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
        tuple of plotly figures: two plotly figures for forecasting and components of forecast
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

    # create a future dataframe to predict the next 56 months
    future = m.make_future_dataframe(periods=56, freq='M')

    # make the prediction
    forecast = m.predict(future)

    # generate the two ploty figures, first showing forecast, second showing forecast components
    fig = plot_plotly(m, forecast)
    fig.update_layout(title="Forecasting Damage over the next 56 Months")
    fig2 = plot_components_plotly(m, forecast)
    fig2.update_layout(title='Forecast Components - Trend & Seasonality')

    return fig, fig2


def _significant_months_barchart(df, metric='damage'):
    """Determines the ten months with the most deaths, injuries or damage from the dataframe given. Uses the metric
    provided to rank months. Generates an ordered bar chart using these ten months.

    Args:
        df (pandas.DataFrame): DataFrame containing monthly sums to analyze
        metric (str, optional): Metric to analyze. Can be 'damage', 'injuries' or 'deaths'. Defaults to 'damage'.

    Returns:
        plotly figure: barchart generated
    """

    # check for erroneous input
    if not metric in ['damage', 'injuries', 'deaths']:
        raise ValueError('Metric must be one of: damage, injuries, deaths')
    elif metric == 'damage': 
        # next few else ifs set the title
        title = 'Costliest Months in History by Damage'
    elif metric == 'deaths':
        title = 'Months in History with Most Deaths caused by NOAA Events'
    else:
        title = 'Months in History with Most Injuries caused by NOAA Events'

    series = df[metric.upper()].nlargest(10) # for the given metric, get the 10 largest values

    # create dataframe for plotting
    data = pd.DataFrame()
    data['Month'] = series.index.strftime("%B %Y") # reformat datetime as MonthName Year
    data['Year'] = series.index.strftime("%Y") # to color by year
    data[metric.upper()] = series.values

    # generate figure using color code
    fig = px.bar(data, x='Month', y=metric.upper(), color='Year', title=title) 
    fig.update_layout(xaxis_categoryorder = 'total descending') # ensure it is descending order

    return fig


def _monthly_dist(df, metric='damage'):
    """Generates the distribution of damage, injuries or deaths by month and produces a plotly bar chart

    Args:
        df (pandas.DataFrame): DataFrame containing monthly sums to analyze
        metric (str, optional): Metric to analyze. Can be 'damage', 'injuries' or 'deaths'. Defaults to 'damage'.

    Returns:
        plotly figure: barchart generated
    """

    if not metric in ['damage', 'injuries', 'deaths']:
        raise ValueError('Metric must be one of: damage, injuries, deaths')
    elif metric == 'damage': 
        # next few else ifs set the title
        title = 'Distribution of Damage over Months'
    elif metric == 'deaths':
        title = 'Distribution of Deaths over Months'
    else:
        title = 'Distribution of Injuries over Months'

    series = df[metric.upper()]

    series = series.groupby(series.index.month).sum()
    series.index = map(lambda m: calendar.month_name[m], series.index)

    # create dataframe for plotting
    data = pd.DataFrame()
    data['Month'] = series.index
    data[metric.upper()] = series.values

    return px.bar(data, x='Month', y=metric.upper(), title=title)