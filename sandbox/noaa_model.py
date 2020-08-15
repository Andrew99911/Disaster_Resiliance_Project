import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.offline as py
py.init_notebook_mode()

d = {'K': 3, 'k': 3, 'M': 6, 'm': 6, 'B': 9, 'b': 9}
def str_to_num(s):
    if isinstance(s, str) and s[-1] in d:
        num, magnitude = s[:-1], s[-1]
        return float(num) * 10 ** d[magnitude]
    elif isinstance(s, str):
        return float(s)
    else:
        return s

state = "MARYLAND"

# read in the excel sheet
df = pd.concat(pd.read_excel('Combined-Data-' + state + '-1998-Apr2020.xlsx', sheet_name=None) , ignore_index=True)

df.reset_index(drop=True, inplace=True) # removing the indices, unnecessary

# set the end date of storm events as the new index
df.set_index('END_DATE_TIME', inplace=True)
df.index = pd.to_datetime(df.index) # convert to datetime objects
df.sort_index(inplace=True) # sort df for viewing/debugging

df = df[['INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 
    'DAMAGE_CROPS',]] # select the important columns

# convert damage columns to integer values
df['DAMAGE_CROPS'] = df['DAMAGE_CROPS'].map(str_to_num)
df['DAMAGE_PROPERTY'] = df['DAMAGE_PROPERTY'].map(str_to_num)

# iterate through all the different metrics that we want to project
for metric in ['DAMAGE_CROPS', 'DAMAGE_PROPERTY']:

    # reformat for the given metric to monthly sums
    series = df[metric].resample('M').sum()
    series = series[np.abs(stats.zscore(series)) < 2] # keep values within two standard deviations

    df_metric = pd.DataFrame()
    df_metric['ds'] = series.index
    df_metric['y'] = series.values

    m = Prophet()
    m.fit(df_metric)

    # create a future dataframe to predict the next 36 months
    future = m.make_future_dataframe(periods=36, freq='M')

    # make the prediction
    forecast = m.predict(future)

    fig = plot_plotly(m, forecast)
    fig2 = plot_components_plotly(m, forecast)
    py.iplot(fig)
    py.iplot(fig2)