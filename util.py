import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.offline as py

d = {'K': 3, 'k': 3, 'M': 6, 'm': 6, 'B': 9, 'b': 9}
def str_to_num(s):
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

def get_df(states):
    """Queries data from the 'data' subdirectory where the state of the row is in the list of states provided as a 
    parameter. The produced dataframe is converted into a dataframe where the injuries, property damange, crop damage, 
    and deaths are saved as columns and indexed by the datetime of that storm.

    Args:
        states (List of str): List of states we want to keep in the dataframe.

    Return:
        pandas.DataFrame: Contains the injuries, deaths, property damage and crop damage of each storm for the states
        given. NOTE: The metrics are resampled to monthly sums.
    """

    return None

