# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#///////////////////////////////////////// Functions /////////////////////////////////////////

def plot_hist(data: list, name: list = [], title: str = '', x_label: str = '', y_label: str ='', bins: int = 0) -> None:
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
            if bins > 0:
                e.hist(bins=bins)
            else:
                e.hist()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def split_data(data: object, columns: list, message: str = '') -> list:
    '''
    Split a data frame by the columns indicated and store it in a list
    '''
    print(f'> {message}')
    result = []
    lim = len(columns)
    for i in range(lim):
        sel = data[data["platform"] == columns[i]]
        print(f'{columns[i]}: {len(sel)}')
        result.append(sel)
    del sel
    return result

def single_correlation(data: object, var: list) -> None:
    '''
    Print the correlation between two variables and plot the data in a scatter plot
    '''
    a = var[0]
    b = var[1]
    
    corr = data[[a, b]].corr()[a][b] 
    print(f'> Corraletion between {a} and {b} is: {corr}')

    data.plot.scatter(x=a, y=b, grid=True, alpha=0.3)
    plt.title(f'Correlation: {corr}')
    plt.show()

def dist_analysis(data: object, objective: str, elements: str, corr_lists: list) -> None:
    '''
    Analysis of data distribution and correlation between multiple variables
    '''
    # Print highest value
    h = data[objective].max()
    print('> The highest value in the data is: ', h)
    
    # Ploting hist
    data[objective].hist()
    plt.title(f'Highest value: {h}')
    plt.show()

    # Does the data need a transformation?
    res = input('> Does the data need a transformation? (y/n): ')

    if res.lower() == 'y':
        data_lg = data.copy()
        data_lg[objective] = np.log(data[objective])
        data = data_lg

    # Sales box plot by platform
    data.boxplot(column=objective, by=elements)
    plt.show()

    # Scatter plots
    for e in corr_lists:
        single_correlation(data, e)

    del data

def reg_analysis(data: object, client: str, objective: str, elements_1: str, elements_2: str, elements_3: str) -> list:
    '''
    Print most popular elements, and other metrics
    '''
    res = [data.groupby(elements_1)[objective].sum().sort_values(ascending = False).head(5)]
    
    res.append(data.groupby(elements_1)[elements_1].value_counts().sort_values(ascending=False).head(5))

    res.append(data.groupby(elements_2)[elements_2].value_counts().sort_values(ascending=False).head(5))

    res.append(data.groupby(elements_3)[objective].sum().sort_values(ascending = False).head())

    for t in res:
        t.name = client

    return res

def sample_mean(data: object, req_data: int, data_size: int, col_name: str) -> object:
    '''
    Sample a data set  mean
    '''
    sample_size = data_size//req_data
    res = []
    print('> Your sample has a size of: ', sample_size)
    if sample_size <= 1:
        print('> Insuficient data')
    else:
        for _ in range(req_data):
            res.append(data.sample(sample_size).mean())
    
    res = pd.DataFrame(res, columns=[col_name])

    return res

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

# Ploting videogame units per platform
plot_hist([data['platform']], title='Total videogames per platform', x_label='Platform', y_label='Units')

# Spliting data by more diverse plataforms
platform = ['DS', 'X360', 'PS3', 'PS2']
result = split_data(data, platform, 'Video games available: ')

# Ploting videogames per platform
platform = ['Nintendo DS', 'Xbox 360', 'Playstation 3', 'Playstation 2']
titles = [f'{c}  titles available per year' for c in platform]
lim = len(platform)
for i in range(lim):
    plot_hist([result[i]['year_of_release']], title=titles[i], x_label='Year', y_label='Units')

# Ploting generetion sales per company
companies = ['Microsoft', 'Sony', 'Nintendo', 'Sony portable', 'Nintendo portable']
platforms = [['XB', 'X360', 'XOne'], ['PS', 'PS2', 'PS3', 'PS4'], ['NES', 'SNES', 'N64', 'GC', 'Wii', 'WiiU'], ['PSP', 'PSV'], ['GB','GBA','DS','3DS']]
lim = len(companies)
plats = {companies[i]:platforms[i] for i in range(lim)}

for c in companies:
    plat = plats[c]
    result = split_data(data, plat, 'Sells: ')
    plot_hist([d['year_of_release'] for d in result], plat, title=f'Videogames available in {c} consoles', x_label='Year', y_label='Units')

# Printing most selled by platform
print('> Best selling by platform')
print(data.groupby('platform')['total_sales'].sum().sort_values(ascending=False))

# New data
data = data[data['year_of_release'] >= 2000]

# Split data by console type, home console (hc) and portable console (pc)
platforms_hc = platforms[:3]
platforms_pc = platforms[3:]

data_hc = data[data['platform'].isin(platforms_hc[0] + platforms_hc[1] + platforms_hc[2])]
data_pc = data[data['platform'].isin(platforms_pc[0] + platforms_pc[1])]

# Analysis
dist_analysis(data_hc, 'total_sales', 'platform', [['critic_score', 'total_sales'], ['user_score', 'total_sales']])

# Same analysis, other data
dist_analysis(data_pc, 'total_sales', 'platform', [['critic_score', 'total_sales'], ['user_score', 'total_sales']])

# Distribution by genre in home consoles
plot_hist([data_hc['genre']], title='Total videogames per genre', x_label='Genre', y_label='Units', bins=len(data_hc['genre'].unique()))

# Best sellers
print('> Best sellers: \n', data_hc[data_hc['total_sales'] > 10])

#///////////////////////////////////////// Regions /////////////////////////////////////////

obj = data_hc[(data_hc['na_sales'] > 0) & (data_hc['eu_sales'] == 0) & (data_hc['jp_sales'] == 0)]
client = 'North America'
res_na = reg_analysis(obj, client, 'total_sales', 'platform', 'genre', 'rating')

obj = data_hc[(data_hc['eu_sales'] > 0) & (data_hc['na_sales'] == 0) & (data_hc['jp_sales'] == 0)]
client = 'Europe'
res_eu = reg_analysis(obj, client, 'total_sales', 'platform', 'genre', 'rating')

obj = data_hc[(data_hc['jp_sales'] > 0) & (data_hc['eu_sales'] == 0) & (data_hc['na_sales'] == 0)]
client = 'Japan'
res_jp = reg_analysis(obj, client, 'total_sales', 'platform', 'genre', 'rating')

titles = ['Sales', 'Platforms', 'Genres', 'Ratings']
lim = len(res_na)
for i in range(lim):
    print(f'\n> {titles[i]}:')
    print(pd.concat([res_na[i], res_eu[i], res_jp[i]], axis=1))

del data_hc, data_pc, res_na, res_eu, res_jp 

#///////////////////////////////////////// Hypothesis test /////////////////////////////////////////

# Testing users scores mean between xbox One and PC
data_xo = data[data['platform'] == 'XOne']
data_pc = data[data['platform'] == 'PC']

# Total data
size_xo = data_xo.shape[0]
size_pc = data_pc.shape[0]
print('> Total Xbox One data:', size_xo)
print('> Total PC data:', size_pc)

# Plot histograms
plot_hist([data_xo['user_score']], title='Xbox One user scores', x_label='Score', y_label='Users')
plot_hist([data_pc['user_score']], title='PC user scores', x_label='Score', y_label='Users')

# Taking samples
sample_xo = sample_mean(data_xo['user_score'], 30, np.min([size_xo, size_pc]) , 'mean_score')
plot_hist([sample_xo])

sample_pc = sample_mean(data_pc['user_score'], 30, np.min([size_xo, size_pc]) , 'mean_score')
plot_hist([sample_pc])