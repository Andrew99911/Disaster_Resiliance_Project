# Util Begins Here

import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import calendar
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import dash
import dash_core_components as dcc
import dash_html_components as html

import datetime

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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

    # use these keys to get the correponding graphs
    ret['prediction'] = _gen_predictive_model(df, use_outliers=True)
    ret['top10'] = _significant_months_barchart(df, metric='damage')
    ret['month_dist'] = _monthly_dist(df, metric='damage')
    ret['total_injuries'] = df['INJURIES'].sum()
    ret['total_deaths'] = df['DEATHS'].sum()
    
    temp = df['DAMAGE'].sum()
    magnitude = 0
    while temp >= 1000:
        magnitude += 1
        temp /= 1000.0
    # add more suffixes if you need them
    ret['total_damage'] =  ('%.2f%s' % (temp, ['', 'K', 'M', 'B', 'T', 'Q'][magnitude]))

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

    if isinstance(s, str) and s[-1] in d and s != '': 
        # if its a string where the final character is K, M, or B, convert accordingly
        num, magnitude = s[:-1], s[-1]
        if len(num) == 0:
            num = 1
        return float(num) * 10 ** d[magnitude]
    elif isinstance(s, str) and s != '': 
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
    
    df = df_total[df_total['STATE'].isin(states)] # only keep data from states specified

    # return the df with unimportant columns droppped and resampled by month and summed (i.e. gives monthly sums instead
    # of individual data points)
    return df.resample('M').sum()


def _gen_predictive_model(df, use_outliers=True):
    """Using the Holt Winters Package, generate two interactive plotly graphs projecting damage for the next 
    56 months (i.e. until 2025). First graph is the historical data, second is the predictions
    Args:
        df (pandas.DataFrame): DataFrame containing monthly sums to analyze/forecast
        use_outliers(Boolean): whether or not to filter out outliers before analysis
    Return:
        tuple of two plotly figures: plotly figures depicting historical data and predictions
    """

    series = df['DAMAGE'] # select the damage series to analyze

    if not use_outliers:
        series = series[np.abs(stats.zscore(series)) < 2] # keep values within two standard deviations

    series.dropna(inplace=True)

    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit() # fit model
    forecast = pd.Series(fit.forecast(56)) # forecast for the next 56 months (to 2025)

    # toss all the original data and predictions into two seperate plotly graphs
    data = pd.DataFrame()
    data['Month'] = series.index
    data['Values'] = series.values
    fig1 = px.line(data, x='Month', y='Values')
    
    data2 = pd.DataFrame()
    data2['Month'] = forecast.index
    data2['Values'] = forecast.values
    fig2 = px.line(data2, x='Month', y='Values')
    
    return fig1, fig2


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
    data['MONTH'] = series.index.strftime("%B %Y") # reformat datetime as MonthName Year
    Decade_Filter = series.index < pd.to_datetime(datetime.date(2009, 1, 1))
    data['TimeFrame'] = ['Before 2009' if x else '2009 & After' for x in Decade_Filter]
    data[metric.upper()] = series.values

    # generate figure using color code
    fig = px.bar(data, x='MONTH', y=metric.upper(), color='TimeFrame', title=title) 
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
    data['MONTH'] = series.index
    data[metric.upper()] = series.values

    return px.line(data, x='MONTH', y=metric.upper(), title=title)

# load total dataframe
dump_sheetnames = ['NOAAStormEventsData-dump1.csv', 'NOAAStormEventsData-dump2.csv']
df_total = pd.concat([pd.read_csv(f, index_col='END_DATE_TIME') for f in dump_sheetnames])
df_total.index = pd.to_datetime(df_total.index)

# Create Initial Graphs
ret = gen_analysis(['MARYLAND'])
figure1, figure2 = ret['prediction']
figure3 = ret['top10']
figure4 = ret['month_dist']
injury = str(ret['total_injuries'])
death = str(ret['total_deaths'])
damage = '$ {}'.format(ret['total_damage'])

regions = pd.read_csv(r'https://raw.githubusercontent.com/Andrew99911/flying-dog-beers/master/Region_List.csv',
                      header=None,error_bad_lines=False)

regions.columns = ['Regions']
regions_drop = []
all_selection = []
for value in regions['Regions']:
    regions_drop.append({'label': value.title(), 'value': value.upper()})
    all_selection.append(value)

app.layout = html.Div(
    style={ 
        'margin': 'auto -5px',
        'width': 'auto',
        'height': '1050px',
        'background-color': 'rgb(245,245,245)',
    },
    children=[
        html.Div(
            children=[
                html.H2(
                    children=[
                        'Modeling Property Damage Across US States & Territories'
                    ],
                    style= {
                        'color':'dark gray',
                        'font-family':'Arvo',
                        'margin-left': '1em',
                        'float':'left'
                    }
                ),
                html.Img(
                    src="https://aeesp.org/sites/default/files/board/UMD-CEE-Faculty-Announcement105678-10-2017.jpg",
                    
                    style = {
                        'float':'right',
                        'height':'80px',
                        'margin-right': '2em',
                    }
                )
            ],
            style = {
                'background-color':'white',
                'border-bottom': '5px solid rgb(207,16,45)',
                'height': '80px'
            }
        ),
        html.Div(
            children = [],
            style = {
                'height':'20px'
            }
        ),
        dcc.Tabs(
            id='Main_tabs', 
            value='tab-1',
            colors = {
                "border": "white",
                "primary": "rgb(255,205,35)",
                "background": "rgb(255,230,144)"
            },
            style= {
                'font-family':'Arvo',
                'font-size':'20px'
            },
            children=[
                dcc.Tab(
                    label='Background', 
                    value='tab-1',
                    children = [
                        html.Div(
                            style={
                                'width': '100%',
                                'height': '20px'
                            } 
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    style={
                                        'width': '5%',
                                        'float':'left',
                                        'height': '600px'
                                    } 
                                ),
                                html.Div(
                                    children = [
                                        html.H4(
                                            children = ['The Problem'],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '0.5em',
                                                 'margin-right': '0.5em'
                                            }
                                        ),
                                        html.H6(
                                            children = [
                                                '''
                                                East coast hurricanes, tornado alley, floods down the Mississippi, and forest fires in the west all demonstrate earth’s ferocious, destructive, and unforgiving nature. 2017’s Hurricane Maria devastated Puerto Rico by destroying 80% of the island’s crops and unplugged the power grid in some areas for a year [1]. Californian wildfires in 2018 alone burnt almost 2 million acres of land or about 11% of their state’s forests [2], and wildfires like these can contribute to around 5-10% of global CO2 emissions [3]. The 2019 Mississippi floods caused 20 billion dollars in damages across several states and resulted in 3 causalities [4].
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',
                                                 'font-size':'16px',
                                                 'margin-right': '1em'
                                            }
                                        ),
                                        html.H4(
                                            children = ['Current Research'],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '0.5em',
                                                 'margin-right': '0.5em'
                                            }
                                        ),  
                                        html.H6(
                                            children = [
                                                '''
                                                These incidents question whether deadlier disasters are here to come or if these examples are sporadic coincidences. Researchers at NASA conclude that these storms will stay and worsen. Hurricanes attain category III speeds 9 hours faster today than 20 years ago, storms deliver extreme downpour 30% more since 1948, and wintery seasons are twice as likely to produce blizzards post 1960 than from the early 1900s [5]. Extreme weather creeps onto unaware citizens and produces growing fatalities, property damage, and crop devastation. 
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'font-size':'16px',
                                                 'margin-right': '1em'
                                            }
                                        ),
                                        html.H4(
                                            children = ['Our Model'],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '0.5em',
                                                 'margin-right': '0.5em'
                                            }
                                        ), 
                                        html.H6(
                                            children = [
                                                '''
                                                We decided to model and predict damages caused by natural disasters as this can showcase the necessity to combat climate change. In order to do this, we first obtained data from the NOAA storm events database across both the continental USA and remote states and territories ranging from 1998 to now [6]. With this data, we programmed a model that can predict fatalities, property damage, and crop damage up to 5 years into the future. Using a package called dash, we hand coded a website and developed ways to interact with our model by showcasing interesting graphs and figures. Scroll through the tabs at the top to view our interactive tools.
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'font-size':'16px',
                                                 'margin-right': '1em'
                                            }
                                        ),
                                        html.H6(
                                            children = ['skip line'],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'margin-right': '1em',
                                                 'color': 'white'
                                            }
                                        ),
                                        html.H6(
                                            children = [
                                                '''
                                                Created in association with University of Maryland's Civil & Enviormental Engineering Department
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1.1em',  
                                                 'font-size':'13px',
                                                 'margin-right': '1em'
                                            }
                                        ),
                                        html.H6(
                                            children = [
                                                '''
                                                Special thanks to Facebook's Prophet, Plotly's Dash, and NOAA for the tools to produce this project
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1.1em',  
                                                 'font-size':'13px',
                                                 'margin-right': '1em'
                                            }
                                        )
                                    ],
                                    style={
                                        'width': '50%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'float':'left',
                                        'height': '750px'
                                    } 
                                ),
                                 html.Div(
                                    style={
                                        'width': '5%',
                                        'float':'left',
                                        'height': '600px'
                                    } 
                                ),
                                html.Div(
                                    children = [
                                        html.Div(
                                            children = [
                                                html.Img(
                                                    src="https://images.unsplash.com/photo-1553984840-ec965a23cddd?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1189&q=80",
                                                style = {
                                                    'width': '100%',
                                                    'height': '350px',
                                                    'border-radius': '10px'
                                                    }
                                                )
                                            ],
                                            style={
                                                'width': '100%',
                                                'border-radius': '10px',
                                                'box-shadow':'3px 3px rgb(200,200,200)',
                                                'background-color': 'white',
                                                'height': '350px'
                                            }
                                        ),
                                        html.Div(
                                            children = [],
                                            style={
                                                'width': '100%',
                                                'height': '20px'
                                            }
                                        ),
                                        html.Div(
                                            children = [ 
                                                html.H6(
                                                    children = ['skip line'],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left': '0.5em',
                                                        'margin-right': '0.5em',
                                                        'color':'white',
                                                        'font-size':'6px'
                                                    }
                                                ),
                                                html.H4(
                                                    children = ['Citations'],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left': '0.5em',
                                                        'margin-right': '0.5em'
                                                    }
                                                ),
                                                html.H6(
                                                    children = [
                                                        '''
                                                        [1] The facts: Hurricane Maria’s effect on Puerto Rico. (2020, January 8). Mercy Corps. https://www.mercycorps.org/blog/quick-facts-hurricane-maria-puerto-rico/
                                                        '''
                                                    ],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left': '1.1em',  
                                                        'font-size':'12px',
                                                        'margin-right': '1em'
                                                    }
                                                ),
                                                html.H6(
                                                    children = [
                                                        '''
                                                         [2] 2018 Fire Season | Welcome to CAL FIRE. (2019). Cal Fire. https://www.fire.ca.gov/incidents/2018/
                                                        '''
                                                    ],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left': '1.1em',  
                                                        'font-size':'12px',
                                                        'margin-right': '1em'
                                                    }
                                                ),
                                                html.H6(
                                                    children = [
                                                        '''
                                                        [3] Berwyn, B. (2018, August 23). How Wildfires Can Affect Climate Change (and Vice Versa). InsideClimate News. https://insideclimatenews.org/news/23082018/extreme-wildfires-climate-change-global-warming-air-pollution-fire-management-black-carbon-co2
                                                        '''
                                                    ],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left': '1.1em',  
                                                        'font-size':'12px',
                                                        'margin-right': '1em'
                                                    }
                                                ),
                                                html.H6(
                                                    children = [
                                                        '''
                                                        [4] Smith, A. (2020, July 8). Billion-Dollar Weather and Climate Disasters: Overview | National Centers for Environmental Information (NCEI). Billion-Dollar Weather and Climate Disasters: Overview. https://www.ncdc.noaa.gov/billions/
                                                        '''
                                                    ],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left': '1.1em',  
                                                        'font-size':'12px',
                                                        'margin-right': '1em'
                                                    }
                                                ),
                                                html.H6(
                                                    children = [
                                                        '''
                                                        [5] In a Warming World, the Storms May Be Fewer but Stronger. (2013, March 5). NASA. https://earthobservatory.nasa.gov/features/ClimateStorms/page2.php
                                                        '''
                                                    ],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left': '1.1em',  
                                                        'font-size':'12px',
                                                        'margin-right': '1em'
                                                    }
                                                ),
                                                html.H6(
                                                    children = [
                                                        '''
                                                        [6] Storm Events Database | National Centers for Environmental Information. (2020). NOAA. https://www.ncdc.noaa.gov/stormevents/
                                                        '''
                                                    ],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left': '1.1em',  
                                                        'font-size':'12px',
                                                        'margin-right': '1em'
                                                    }
                                                ),
                                                
                                            ],
                                            style={
                                                'width': '100%',
                                                'border-radius': '10px',
                                                'box-shadow':'3px 3px rgb(200,200,200)',
                                                'background-color': 'white',
                                                'height': '370px'
                                            }
                                        )
                                     ],
                                     style={
                                        'width': '35%',
                                        'float':'left',
                                        'height': '600px'
                                    }
                                )
                            ],
                            style={
                                'width': '100%',
                                'height': '600px'
                            } 
                        ),
                        
                    ]
                ),
                dcc.Tab(
                    label='How to Use', 
                    value='tab-2',
                    children = [
                        html.Div(
                            children = [],
                            style={
                                'width': '100%',
                                'height': '20px'
                            } 
                        ),
                        html.Div(
                            children = [
                                html.Div(
                                    children = [],
                                    style={
                                        'width': '5%',
                                        'height': '400px',
                                        'float':'left'
                                    } 
                                ),
                                html.Div(
                                    children = [
                                        html.H4(
                                            children = ['How to Use this Site'],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '0.5em',
                                                 'margin-right': '0.5em'
                                            }
                                        ), 
                                        html.H6(
                                            children = [
                                                '''
                                                We encourage users to let their curiosity do the searching and our app to do the heavy work. If you’re interested in decade to decade trends across the entire USA, head over to the topographical display to filter by various damage indicators and see their change across states and territories. Perhaps you’re curious about your specific state, you can find its worst storms, average damage by month, and a predictive future model in the graphical display.
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'font-size':'16px',
                                                 'margin-right': '1em'
                                            }
                                        ),
                                        html.H6(
                                            children = 'skip line',
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'font-size':'16px',
                                                 'margin-right': '1em',
                                                 'color':'white'
                                            }
                                        ),
                                        html.H6(
                                            children = [
                                                '''
                                                On the right you can see our recommended way to use the app. First, we use our topographic map to investigate general trends in an area. Second, we select a region of interest (Maryland), and go to our graphical display to see trends on it. Finally, we use the internet to explain the trends that we see. Maryland has seen decreasing property damages because Hurricane Isabell contributed to the majority of their post-1998 weather damages. Now that we know, how can we interrupt this? (See Below)                                          
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'font-size':'16px',
                                                 'margin-right': '1em'
                                            }
                                        ),
                                    ],
                                    style={
                                        'width': '50%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '400px',
                                        'float':'left'
                                    },
                                ),
                                html.Div(
                                    children = [],
                                    style={
                                        'width': '5%',
                                        'height': '400px',
                                        'float':'left'
                                    } 
                                ),
                                html.Div(
                                    children = [
                                        html.Img(
                                            src="https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/Howto.PNG",
                                            style = {
                                                'width': '100%',
                                                'height': '400px',
                                                'border-radius': '10px'
                                            }
                                        )    
                                    ],
                                    style={
                                        'width': '35%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '400px',
                                        'float':'left'
                                    },
                                ),
                                html.Div(
                                    children = [],
                                    style={
                                        'width': '5%',
                                        'height': '400px',
                                        'float':'left'
                                    } 
                                ),
                            ],
                            style = {
                                'width': '100%',
                                'height': '400px'
                            }
                        ),
                        html.Div(
                            children = [],
                            style={
                                'width': '100%',
                                'height': '20px'
                            } 
                        ),
                        html.Div(
                            children = [
                                html.Div(
                                    children = [],
                                    style={
                                        'width': '5%',
                                        'height': '400px',
                                        'float':'left'
                                    } 
                                ),
                                html.Div(
                                    children = [
                                        html.Img(
                                            src=r"https://www.netclipart.com/pp/m/23-236444_changing-to-night-clipart-perched-climate-change-clip.png",
                                            style = {
                                                'width': '100%',
                                                'height': '400px',
                                                'border-radius': '10px'
                                            }
                                        )     
                                    ],
                                    style={
                                        'width': '35%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '400px',
                                        'float':'left'
                                    },
                                ),
                                html.Div(
                                    children = [],
                                    style={
                                        'width': '5%',
                                        'height': '400px',
                                        'float':'left'
                                    } 
                                ),
                                html.Div(
                                    children = [
                                        html.H4(
                                            children = ['Suggestions to Interpret'],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '0.5em',
                                                 'margin-right': '0.5em'
                                            }
                                        ), 
                                        html.H6(
                                            children = [
                                                '''
                                                Our data can visualize lots of trends, but it’s important to do research whether overarching conclusions can be drawn from what you’re investigating. For example, every state that starts with the letter ‘M’ might have increasing weather induced damages, but concluding that climate change is targeting ‘M’ lettered states would be silly. Be sure to research the trends you find so that they are valid.
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'font-size':'16px',
                                                 'margin-right': '1em'
                                            }
                                        ),
                                        html.H6(
                                            children = 'skip line',
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'font-size':'16px',
                                                 'margin-right': '1em',
                                                 'color':'white'
                                            }
                                        ),
                                        html.H6(
                                            children = [
                                                '''
                                                Research groups like NASA confirmed that climate change doesn’t produce extreme weather in every corner of the earth; instead, climate change will bring stronger storms in some areas and milder storms in others. The reason climate change is a problem is because it overall is set to produce worse conditions. On the topographical display, you may see that weather damages have increased for nearly every category from decade to decade. Climate change is very complex, and our tool was built to visualize it, but does not have the ability to fully explain it.
                                                '''
                                            ],
                                            style = {
                                                'font-family':'Arvo',
                                                 'margin-left': '1em',  
                                                 'font-size':'16px',
                                                 'margin-right': '1em'
                                            }
                                        ),
                                    ],
                                    style={
                                        'width': '50%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '400px',
                                        'float':'left'
                                    },
                                ),
                                html.Div(
                                    children = [],
                                    style={
                                        'width': '5%',
                                        'height': '400px',
                                        'float':'left'
                                    } 
                                ),
                            ],
                            style = {
                                'width': '100%',
                                'height': '400px'
                            }
                        ),
                    ]
                ),
                dcc.Tab(
                    label='Topographical Display', 
                    value='tab-3',
                    children = [
                        html.Div(
                            children = [],
                            style={
                                'width': '100%',
                                'height': '20px'
                            } 
                        ),
                        html.Div(
                            children = [
                                html.Div(
                                    children = [],
                                    style={
                                        'width': '5%',
                                        'height': '600px',
                                        'float':'left'
                                    } 
                                ),
                                html.Div(
                                    children = [
                                        html.Div(
                                            children = [
                                                html.Div(
                                                    children = [
                                                        html.H6(
                                                            children = 'Skip Line',
                                                            style = {
                                                                    'font-family':'Arvo',
                                                                    'font-size': '6px',
                                                                    'color':'white'
                                                                }
                                                        ), 
                                                        html.H4(
                                                        children = 'Select Map Display',
                                                        style = {
                                                                'font-family':'Arvo',
                                                            }
                                                        ),
                                                        html.H6(
                                                            children = 'Select a variable to see trends by state:',
                                                            style = {
                                                                    'font-family':'Arvo',
                                                                    'font-size': '16px',
                                                                }
                                                        ), 
                                                        dcc.Dropdown(
                                                            id='map_dropdown',
                                                            options=[
                                                                {'label': 'By Injuries', 'value': 'Injury'},
                                                                {'label': 'By Fatalities', 'value': 'Death'},
                                                                {'label': 'By Property Damage', 'value': 'Property'},
                                                                {'label': 'By Crop Damage', 'value': 'Crop'},
                                                                {'label': 'By Total Damage', 'value': 'Total'},
                                                            ],
                                                            value='Total',
                                                        ),
                                                        html.H6(
                                                            children = 'Skip Line',
                                                            style = {
                                                                    'font-family':'Arvo',
                                                                    'font-size': '18px',
                                                                    'color':'white'
                                                                }
                                                        ), 
                                                        html.H6(
                                                            children = 'Overall is changing by',
                                                            style = {
                                                                    'font-family':'Arvo',
                                                                    'font-size': '16px',
                                                                    'color':'black'
                                                                }
                                                        ), 
                                                        html.H6(
                                                            id = 'map-percent-change',
                                                            children = '+8.1% each Decade',
                                                            style = {
                                                                    'font-family':'Arvo',
                                                                    'font-size': '18px',
                                                                    'color':'red'
                                                                }
                                                        ),
                                                    ],
                                                    style = {
                                                        'margin-left':'5%',
                                                        'width':'90%'
                                                    }
                                                ),
                                            ],
                                            style={
                                                'width': '100%',
                                                'border-radius': '10px',
                                                'box-shadow':'3px 3px rgb(200,200,200)',
                                                'background-color': 'white',
                                                'height': '270px',
                                            },
                                        ),
                                        html.Div(
                                            style={
                                                'width': '100%',
                                                'height': '20px',
                                            },
                                        ),
                                        html.Div(
                                            children = [
                                                html.H6(
                                                    children = 'Skip Line',
                                                    style = {
                                                            'font-family':'Arvo',
                                                            'margin-left': '0.5em',
                                                            'margin-right': '0.5em',
                                                            'font-size': '6px',
                                                            'color':'white'
                                                        }
                                                ), 
                                                html.H4(
                                                    children = 'Legend for Map',
                                                    style = {
                                                            'font-family':'Arvo',
                                                            'margin-left': '0.5em',
                                                            'margin-right': '0.5em'
                                                        }
                                                ),                                                
                                                html.Img(
                                                    src=r"https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/maps/Legend.PNG?token=ANRXGELY6NNOLTEBGTRNBPK7HAWKK",                                            
                                                    style = {
                                                        'margin-left':'5%',
                                                        'height':'200px',
                                                        'width': '90%',
                                                        'border-radius': '10px',
                                                    }
                                                )    
                                            ],
                                            style={
                                                'width': '100%',
                                                'border-radius': '10px',
                                                'box-shadow':'3px 3px rgb(200,200,200)',
                                                'background-color': 'white',
                                                'height': '300px',
                                            },
                                        ),
                                    ],
                                    style={
                                        'width': '20%',
                                        'height': '600px',
                                        'float':'left'
                                    },
                                ),
                                html.Div(
                                    children = [],
                                    style={
                                        'width': '5%',
                                        'height': '600px',
                                        'float':'left'
                                    } 
                                ),
                                html.Div(
                                    children = [
                                        html.Img(
                                            id = 'the-map',
                                            src=r"https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/maps/Map_Damage.png",                                            
                                            style = {
                                                'height':'600px',
                                                'width': '100%',
                                                'border-radius': '10px',
                                            }
                                        )  
                                    ],
                                    style={
                                        'width': '65%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '600px',
                                        'float':'left'
                                    },
                                ),
                            ],
                            style={
                                'width': '100%',
                                'height': '600px'
                            } 
                        ),
                    ]
                ),
                dcc.Tab(
                    label='Graphical Display', 
                    value='tab-4',
                    children = [
                        html.Div(
                            style={
                                'height':'20px',
                                'width': '100%'
                            }
                        ),
                        # This is the first row (data filter & first graph)
                        html.Div(
                            children=[
                                html.Div(
                                    style={
                                        'height':'20px',
                                        'width': '5%',
                                        'float':'left'
                                    }
                                ),
                                html.Div(
                                    id = 'filter_container',
                                    children=[
                                        html.Div(
                                            children= [
                                                html.H6(
                                                    children = 'Specify a Past or Future Model',
                                                    style = {
                                                        'font-family':'Arvo',
                                                    }
                                                ),
                                                dcc.RadioItems(
                                                   id='radio_graph',
                                                    options=[
                                                        {'label': 'Show Historical Trends', 'value': 'Past'},
                                                        {'label': 'Show Predictive Trends', 'value': 'Future'},
                                                    ],
                                                    value='Past'
                                                ),
                                                html.H6(
                                                    children = 'Filter the data with a region:',
                                                    style = {
                                                        'font-family':'Arvo',
                                                    }
                                                ),
                                                html.Button(
                                                    'Select All', 
                                                    id='submit-val', 
                                                    n_clicks=0
                                                ), 
                                                dcc.Dropdown(
                                                    id='region_dropdown',
                                                    options=regions_drop,
                                                    value='MARYLAND',
                                                    multi=True,
                                                ),
                                            ],
                                            style = {
                                                'margin-left': '3%',
                                                'width': '94%'
                                            }
                                        )
                                    ],
                                    style={
                                        'width': '30%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'float':'left',
                                        'height': '670px'
                                    } 
                                ),
                                html.Div(
                                    style={
                                        'height':'20px',
                                        'width': '5%',
                                        'float':'left'
                                    }
                                ),
                                html.Div(
                                    children = [
                                        html.Div(
                                            id = 'graph1_container',
                                            children=[
                                                html.H6(
                                                    children = ['Skip Line'],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left':'1em',
                                                        'color':'white',
                                                        'font-size':'12px'
                                                    }
                                                ),
                                                html.H4(
                                                    children = ['Predicted Future Damage'],
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'margin-left':'1em'
                                                    }
                                                ),
                                                dcc.Graph(
                                                    id='graph-1',
                                                    figure=figure1,
                                                    style = {
                                                        'height':'460px'
                                                    }
                                                )
                                            ],
                                            style={
                                                'width': '100%',
                                                'border-radius': '10px',
                                                'box-shadow':'3px 3px rgb(200,200,200)',
                                                'background-color': 'white',
                                                'height': '550px',
                                                'float':'left'
                                            }
                                        ),
                                        html.Div(
                                            style={
                                                'height':'570px',
                                                'width': '100%',
                                            }
                                        ),
                                        html.Div(
                                            children = [
                                                html.H6(
                                                    children = 'Skip Line',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'color':'white',
                                                        'font-size':'1px'
                                                    }
                                                ),
                                                html.H5(
                                                    children = 'Total Injuries',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'text-align': 'center'
                                                    }
                                                ),
                                                html.H5(
                                                    children = injury,
                                                    id = 'injuries_text',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'text-align': 'center'
                                                    }
                                                ),                                                
                                            ],
                                            style={
                                                'width': '30%',
                                                'border-radius': '10px',
                                                'box-shadow':'3px 3px rgb(200,200,200)',
                                                'background-color': 'white',
                                                'height': '100px',
                                                'float':'left'
                                            }
                                        ),
                                        html.Div(
                                            style={
                                                'height':'200px',
                                                'width': '5%',
                                                'float': 'left'
                                            }
                                        ),
                                        html.Div(
                                            children = [
                                                html.H6(
                                                    children = 'Skip Line',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'color':'white',
                                                        'font-size':'1px'
                                                    }
                                                ),
                                                html.H5(
                                                    children = 'Total Fatalities',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'text-align': 'center'
                                                    }
                                                ),
                                                html.H5(
                                                    children = death,
                                                    id = 'death_text',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'text-align': 'center'
                                                    }
                                                ),       
                                            ],
                                            style={
                                                'width': '30%',
                                                'border-radius': '10px',
                                                'box-shadow':'3px 3px rgb(200,200,200)',
                                                'background-color': 'white',
                                                'height': '100px',
                                                'float':'left'
                                            }
                                        ),
                                        html.Div(
                                            style={
                                                'height':'200px',
                                                'width': '5%',
                                                'float': 'left'
                                            }
                                        ),
                                        html.Div(
                                            children = [
                                                html.H6(
                                                    children = 'Skip Line',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'color':'white',
                                                        'font-size':'1px'
                                                    }
                                                ),
                                                html.H5(
                                                    children = 'Total Damage',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'text-align': 'center'
                                                    }
                                                ),
                                                html.H5(
                                                    children = damage,
                                                    id = 'total_text',
                                                    style = {
                                                        'font-family':'Arvo',
                                                        'text-align': 'center'
                                                    }
                                                ),       
                                            ],
                                            style={
                                                'width': '30%',
                                                'border-radius': '10px',
                                                'box-shadow':'3px 3px rgb(200,200,200)',
                                                'background-color': 'white',
                                                'height': '100px',
                                                'float':'left'
                                            }
                                        )
                                    ],
                                    style={
                                        'width': '55%',
                                        'height': '670px',
                                        'float':'left'
                                    }
                                ),
                                html.Div(
                                    style={
                                        'height':'20px',
                                        'width': '5%',
                                        'float':'left'
                                    }
                                ),
                            ]
                        ),
                        # This is the second row 
                        html.Div(
                            style={
                                'height':'690px',
                                'width': '100%',
                                'background-color':'rgb(245,245,245)'
                            }
                        ),
                        html.Div(
                            children = [
                                html.Div(
                                    style={
                                        'height':'20px',
                                        'width': '5%',
                                        'float': 'left'
                                    }
                                ),
                                html.Div(
                                    children = [
                                        dcc.Graph(
                                            id='graph-2',
                                            figure=figure3,
                                            style = {
                                                'height':'370px'
                                            }
                                        )    
                                    ],
                                     style={
                                        'width': '42.5%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '400px',
                                        'float':'left'
                                    }
                                ),
                                html.Div(
                                    style={
                                        'height':'400px',
                                        'width': '5%',
                                        'float':'left'
                                    }
                                ),
                                html.Div(
                                    children = [
                                        dcc.Graph(
                                            id='2',
                                            figure=figure4,
                                            style = {
                                                'height':'370px'
                                            }
                                        )    
                                    ],
                                    style={
                                        'width': '42.5%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '400px',
                                        'float':'left'
                                    }
                                ),
                                html.Div(
                                    style={
                                        'height':'400px',
                                        'width': '5%',
                                        'float':'left'
                                    }
                                ),
                            ],
                            style={
                                'height':'500px',
                                'width': '100%',
                                'background-color':'rgb(245,245,245)'
                            }
                        ),
                    ],
                )
            ]
        )
    ]
)

@app.callback(
    [dash.dependencies.Output('map-percent-change', 'children'),
     dash.dependencies.Output('map-percent-change', 'style'),
     dash.dependencies.Output('the-map', 'src')],
    [dash.dependencies.Input('map_dropdown', 'value')])

def change_map_descriptor(input1):
    
    if input1 == 'Injury':
        return ['-0.6% each Decade',{'font-family':'Arvo','font-size': '18px','color':'green'},r"https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/maps/Map_Injuries.png"]
    elif input1 == 'Death':
        return ['+17.1% each Decade',{'font-family':'Arvo','font-size': '18px','color':'red'},r"https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/maps/Map_Deaths.png"]
    elif input1 == 'Property':
        return ['+8.3% each Decade',{'font-family':'Arvo','font-size': '18px','color':'red'},r"https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/maps/Map_PropertyDamage.png"]
    elif input1 == 'Crop':
        return ['+6.1% each Decade',{'font-family':'Arvo','font-size': '18px','color':'red'},r"https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/maps/Map_CropDamage.png"]
    elif input1 == 'Total':
        return ['+8.1% each Decade',{'font-family':'Arvo','font-size': '18px','color':'red'},r"https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/maps/Map_Damage.png"]
    else:
        return ['Select an Area',{'font-family':'Arvo','font-size': '18px','color':'black'},r"https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/maps/Map_Damage.png"]

@app.callback(
    dash.dependencies.Output('region_dropdown', 'value'),
    [dash.dependencies.Input('submit-val', 'n_clicks')])
    
def choose_all(input1):
    if input1 > 0:
        return all_selection
    else:
        return ['MARYLAND']

@app.callback(
    [dash.dependencies.Output('graph-1', 'figure'),
     dash.dependencies.Output('graph-2', 'figure'),
     dash.dependencies.Output('graph-3', 'figure'),
     dash.dependencies.Output('injuries_text', 'children'),
     dash.dependencies.Output('death_text', 'children'),
     dash.dependencies.Output('total_text', 'children'),],
    [dash.dependencies.Input('region_dropdown', 'value'),
     dash.dependencies.Input('radio_graph', 'value')])

def change_graphs(input1,input2):
    
    print(input1)
    
    if input1 == []:
        input1 = ['MARYLAND']
    
    new_ret = gen_analysis(input1)
    new_figure1,new_figure2 = new_ret['prediction']
    new_figure3 = new_ret['top10']
    new_figure4 = new_ret['month_dist']
    
    new_injury = str(new_ret['total_injuries'])
    new_death = str(new_ret['total_deaths'])
    new_damage = '$ {}'.format(new_ret['total_damage'])

    if input2 == 'Future':
        new_figure = new_figure2
    else: 
        new_figure = new_figure1
    
    return [new_figure,new_figure3,new_figure4,new_injury,new_damage,new_death]

if __name__ == '__main__':
    app.run_server(debug=False)