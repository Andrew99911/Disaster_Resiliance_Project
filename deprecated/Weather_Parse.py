# Import Modules
import os
from wwo_hist import retrieve_hist_data

# Specify Conditions
frequency = 24
start_date = '1-APR-2010'
end_date = '20-JUL-2020'

# Subscription Key (60 Day Free Trial)
api_key = '2f8ad339efba4e0eac910040203107'

# Directory to Save Data
os.chdir(r'C:\Users\Andrew\OneDrive\Documents\Disaster Project')

# Location Zip Code
location_list = ['21043']

# Parsing Job
hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)
