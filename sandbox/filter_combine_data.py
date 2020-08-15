"""
Script will create a single spreadsheet of NOAA Storm Events data by concantenating each year's NOAA data. Gets the 
spreadsheets from the data sub-directory (assumed that only sheets with relevant data are there). Outputs to a combined 
spreadsheet.

NOTE: Modified to filter only for a given value to a given column
"""

import pandas as pd
import os

# change into the data directory
subdir = "data"
filter_column = 'STATE'
filter_value = 'MARYLAND'

# gets list of spreadsheets (NOTE: its assumed that data directory only contains relevant spreadsheets)
sheets_paths = os.listdir(subdir) 

with pd.ExcelWriter('Combined-Data-' + filter_value + '-1998-Apr2020.xlsx') as writer:
    for f in sheets_paths:
        df = pd.read_csv(os.path.join(subdir, f))
        df = df.loc[df[filter_column] == filter_value]
        df.to_excel(writer, sheet_name=f[30:34])
        print(f[30:34], ' completed')