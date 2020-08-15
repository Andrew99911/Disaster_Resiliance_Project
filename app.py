import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

regions = pd.read_csv(r'https://raw.githubusercontent.com/Andrew99911/Disaster_Resiliance_Project/master/data/Region_List.csv?token=ANRXGENBKEJ3ZHX6QOD56KK7G5V6O',header=None)
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
        'height': '1000px',
        'background-color': 'rgb(245,245,245)',
    },
    children=[
        html.Div(
            children=[
                html.H2(
                    children=[
                        'Modelling Property Damage Across US States & Territories'
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
                    label='Introduction', 
                    value='tab-1',
                ),
                dcc.Tab(
                    label='How to Use', 
                    value='tab-2',
                ),
                dcc.Tab(
                    label='Interactive Tool', 
                    value='tab-3',
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
                                    children=[
                                        html.Div(
                                            children= [
                                                html.H6(
                                                    children = 'Filter the data with a date range:',
                                                    style = {
                                                        'font-family':'Arvo',
                                                    }
                                                ),
                                                dcc.RangeSlider(
                                                   id='date_slider',
                                                    marks={
                                                        1998: '1998',
                                                        2007: '2007',
                                                        2016: '2016',
                                                        2025: '2025'
                                                    },
                                                    min=1998,
                                                    max=2025,
                                                    step=0.5,
                                                    value=[1998, 2025],
                                                    allowCross=False,
                                                ),
                                                html.H6(
                                                    children = 'Filter the data with a region:',
                                                    style = {
                                                        'font-family':'Arvo',
                                                    }
                                                ),
                                                dcc.Checklist(
                                                    id='region_checklist',
                                                    options=[
                                                        {'label': 'Select All', 'value': 'All'},
                                                    ],
                                                    value=[],
                                                    labelStyle={'display': 'inline-block'}
                                                ), 
                                                dcc.Dropdown(
                                                    id='region_dropdown',
                                                    options=regions_drop,
                                                    value='MARYLAND',
                                                    multi=True,
                                                ),
                                            ],
                                            style = {
                                                'margin-left': '5%',
                                                'width': '90%'
                                            }
                                        )
                                    ],
                                    style={
                                        'width': '30%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '500px',
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
                                html.Div(
                                    children=[
                                        html.H5(
                                            children = [],
                                            style = {
                                                'margin-left':'1em',
                                                'font-family':'Arvo'
                                            }
                                        )
                                    ],
                                    style={
                                        'width': '55%',
                                        'border-radius': '10px',
                                        'box-shadow':'3px 3px rgb(200,200,200)',
                                        'background-color': 'white',
                                        'height': '500px',
                                        'float':'left'
                                    }
                                ),
                                html.Div(
                                    style={
                                        'height':'20px',
                                        'width': '5%',
                                        'float':'left'
                                    }
                                )
                            ]
                        )  
                    ],
                )
            ]
        )
    ]
)

@app.callback(
    dash.dependencies.Output('region_dropdown', 'value'),
    [dash.dependencies.Input('region_checklist', 'value')])

def update_output(input1):
    if input1 == []:      
        return []
    elif input1 == ['All']:
        return all_selection

if __name__ == '__main__':
    app.run_server()
