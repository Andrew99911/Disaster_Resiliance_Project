import csv
import pandas as pd
import os

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
        return 0

# Script to calculate percent change 

# get list of territories
with open('data/Region_list.csv') as f:
    reader = csv.reader(f)
    states = list(reader)

states = [elem[0] for elem in states]

# columns to keep
columns = ['STATE', 'END_DATE_TIME', 'INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT', 
    'DAMAGE_PROPERTY', 'DAMAGE_CROPS',]

sheets_paths = os.listdir('data')

df = pd.concat([pd.read_csv(os.path.join('data', f), usecols=columns) for f in sheets_paths[1:]],ignore_index=True)

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

df.drop(columns=['INJURIES_DIRECT','INJURIES_INDIRECT','DEATHS_DIRECT','DEATHS_INDIRECT'], inplace=True)

# change = {}

# for state in states:
#     df_state = df[df['STATE'] == state]
#     df_state.drop(columns=['STATE'], inplace=True)

#     decade_2000 = df_state[(df_state.index >= '2000-1-1') & (df_state.index < '2010-1-1')].sum()
#     decade_2010 = df_state[(df_state.index >= '2010-1-1') & (df_state.index < '2020-1-1')].sum()
    
#     temp = ( decade_2010 - decade_2000 ) / decade_2000 * 100
#     change[state] = temp.tolist()

# a = pd.DataFrame.from_dict(change, orient='index', columns=['Damage Property', 'Damage Crops', 'Injuries', 'Deaths', 'Damage'])
# a.to_csv('percent_change.csv')

decade_2000 = df[(df.index >= '2000-1-1') & (df.index < '2010-1-1')].sum()
decade_2010 = df[(df.index >= '2010-1-1') & (df.index < '2020-1-1')].sum()

temp = ( decade_2010 - decade_2000 ) / decade_2000 * 100
print(temp)