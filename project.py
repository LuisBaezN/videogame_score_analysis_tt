# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#///////////////////////////////////////// Functions /////////////////////////////////////////

def plot_hist(data: list, name: list = [''], title: str = '', x_label: str = '', y_label: str ='') -> None:
    '''
    Plot an histogram.
    '''
    lim = len(data)
    if lim > 0 and len(name) == lim:
        inc = 0.5 / lim
        a = 1
        i = 0
        for e in data:
            e.hist(alpha=a-i)
            i += inc
            plt.legend(name)
    else:
        for e in data:
            e.hist()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

#///////////////////////////////////////// Initialization /////////////////////////////////////////

# Loading data
path = 'datasets/games.csv'
data = pd.read_csv(path)

#///////////////////////////////////////// Data cleaning /////////////////////////////////////////

# Column names vizualization
print('> Column names: ', data.columns)

# Changing column names to lower case
data.columns = data.columns.str.lower()

# Duplicated data verification
print('> Number of rows duplicated: ', data.duplicated().sum())

# Type visualization
print('> Data vizualization: ')
data.info()

# Changing types
data['year_of_release'] = data['year_of_release'].astype('Int64')
data['critic_score'] = data['critic_score'].astype('Int64')
data['user_score'] = data['user_score'].replace('tbd', None).astype('float')

# Replacing missing data
data['name'] = data['name'].fillna('unknown') # the only 2 that also doesn't have a genre
data['genre'] = data['genre'].fillna('unknown') 
data['rating'] = data['rating'].fillna('unknown')

# Adding data
data['total_sales'] = data['na_sales'] + data['eu_sales'] + data['jp_sales'] + data['other_sales']

#///////////////////////////////////////// EDA /////////////////////////////////////////

# Ploting year of release
plot_hist([data['year_of_release']], title='Units released per year', x_label='Year', y_label='Units')

# Ploting sales per platform
plot_hist([data['platform']], title='Total sales per platform', x_label='Platform', y_label='Units')

# Spliting data by best selling plataforms (DS, X360, PS3, PS2)
b_sellers = ['DS', 'X360', 'PS3', 'PS2']

# Best seller by platform
print('> Best sellers:')
result = []
for i in range(4):
    sel = data[data["platform"] == b_sellers[i]]
    print(f'{b_sellers[i]}: {len(sel)}')
    result.append(sel)

data_ds, data_x360, data_ps3, data_ps2 = result
del result, sel

# Ploting sales per platform
plot_hist([data_ds['year_of_release']], title='Nintendo DS sales per year', x_label='Year', y_label='Units')
plot_hist([data_x360['year_of_release']], title='Xbox 360 sales per year', x_label='Year', y_label='Units')
plot_hist([data_ps2['year_of_release']], title='Play Station 2 sales per year', x_label='Year', y_label='Units')
plot_hist([data_ps3['year_of_release']], title='Play Station 3 sales per year', x_label='Year', y_label='Units')

plot_hist([data_ps2['year_of_release'], data_ps3['year_of_release']], ['PS2', 'PS3'])