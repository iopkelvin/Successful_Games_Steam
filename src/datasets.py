import pandas as pd
import datetime
import numpy as np

path = '/Users/kelvin/'

#Dropped Datasets
# ## GOOGLE PLAY
# google_play = pd.read_csv(path + 'GAMES/data/google-play-store-apps/googleplaystore.csv')
# google_play.drop(['Android Ver', 'Current Ver'], axis=1, inplace=True)
# # print("['Columns dropped', 'Android Ver', 'Current Ver']")
# # print("Google Play dataset: ", google_play.shape)
#
# ## APP STORE
# app_store = pd.read_csv(path + 'GAMES/data/appstore_games.csv')
# app_store.drop(['Subtitle' ,'ID', 'URL', 'Icon URL', 'Description', 'Developer', 'Languages'], axis=1, inplace=True)
# # print("Columns dropped: ['Subtitle' ,'ID', 'URL', 'Icon URL', 'Description', 'Developer', 'Languages']")
# # print("App Store dataset 2016: ", app_store.shape)
# app_store = app_store[app_store['Primary Genre']=='Games']
# # print("App Store dataset 2016 ONLY games: ", app_store.shape)
#
# ## PAN EUROPEAN GAME INFO
# pegi_rating = pd.read_csv(path + 'GAMES/data/PEGI_Ratings_20170907.csv')
# # print("PEGI dataset: ", pegi_rating.shape)
#
# ## VIDEO GAME SALES 2016
# ### PS4 SALES
# vg_sales2016_ps4 = pd.read_csv(path + 'GAMES/data/games-sales/PS4_GamesSales.csv', sep=',', engine='python')
# # print('VG Sales 2016 PS4 dataset: ', vg_sales2016_ps4.shape)
# ### XBOX SALES
# vg_sales2016_xbox = pd.read_csv(path + 'GAMES/data/games-sales/XboxOne_GameSales.csv', sep=',', engine='python')
# # print('VG Sales 2016 XBOX dataset: ', vg_sales2016_xbox.shape)
# ### TOTAL SALES
# vg_sales2016 = pd.read_csv(path + 'GAMES/data/games-sales/Video_Games_Sales_as_at_22_Dec_2016.csv')
# vg_sales2016.drop(['Publisher', 'Developer', 'User_Count', 'Critic_Count'], axis=1, inplace=True)
# # print('VG Sales 2016 dataset: ', vg_sales2016.shape)
#
# ## VIDEO GAMES 2019
# vg_sales2019 = pd.read_csv(path + 'GAMES/data/vgsales-12-4-2019-short.csv')
# vg_sales2019.drop(['Rank', 'Publisher', 'Developer', 'Global_Sales'], axis=1, inplace=True)
# # print("Columns dropped: ['Rank', 'Publisher', 'Developer', 'Global_Sales']")
# # print("VG Sales dataset: ", vg_sales2019.shape)
#
# ## GAMES JSON-FORMAT
# games_json = pd.read_json(path + 'GAMES/data/games.json')
# games_json.drop(['url', 'oses', 'developer', 'cloud_saves', 'controller_support', 'overlay', 'single_player', 'achievement', 'multi_player', 'coop', 'leaderboard', 'in_development', 'languages', 'achievements', 'reviews'], axis=1, inplace=True)
# print("columns dropped: ['url', 'oses', 'developer', 'cloud_saves', 'controller_support', 'overlay', 'single_player', 'achievement', 'multi_player', 'coop', 'leaderboard', 'in_development', 'languages', 'achievements', 'reviews']")
# print("Games Json Dataset: ", games_json.shape)

## STEAM 200K
steam = pd.read_csv(path + 'GAMES/data/steam-200k.csv', names=['customer_id', 'game', 'status', 'hours', 'owned'])
# print("Steam database: ", steam.shape)

## STEAM STORE
### MAIN
steam_store = pd.read_csv(path + 'GAMES/data/steam-store-games/steam.csv')
#steam_store.drop(['english', 'developer', 'publisher', 'categories', 'steamspy_tags', 'achievements'], axis=1, inplace=True)
# print("Columns dropped: ['english', 'developer', 'publisher', 'categories', 'steamspy_tags', 'achievements']")
# print("Steam Store Dataset: ", steam_store.shape)
### CHARACTERISTICS
steam_store_spy = pd.read_csv(path + 'GAMES/data/steam-store-games/steamspy_tag_data.csv')
# print("Steam Store Spy Dataset: ", steam_store_spy.shape)
#Feature dataset STEAM
steam_features = pd.read_csv(path + 'GAMES/data/games-features.csv').rename(columns={'QueryID':'appid'})
# print(steam_features.shape)

## COMBINED
steam_combined = steam_store.merge(steam_store_spy, how='left', on='appid')
## COMBINED + FEATURES
steam_combined_features = steam_combined.merge(steam_features, how='left', on='appid')

#### STEAM COMBINED

df = steam_combined

######## RELEASE DATE ########
release_datetime = pd.to_datetime(df['release_date'])
## Year and month of release
release_year = release_datetime.dt.year.rename('release_year')
release_month = release_datetime.dt.month.rename('release_month')
release_day = release_datetime.dt.day.rename('release_day')
release_weekday = release_datetime.dt.day_name().rename('release_weekday')
## CONCAT
release_dates = pd.concat([release_year, release_month, release_day, release_weekday], axis=1, )
## Days since release
difference_date = datetime.datetime.now() - release_datetime
days_since_release = difference_date.dt.days
release_dates['days_since'] = days_since_release
## CYCLICAL DATA
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
## Encoding month
release_dates = encode(release_dates, 'release_month', 12)
## Encoding day
release_dates = encode(release_dates, 'release_day', 31)
## Years since
years_since = release_dates.release_year.rank(method='dense', ascending=False)
release_dates['release_years_ago'] = years_since
#dummies for days of week
weekday_dummies = pd.get_dummies(release_dates["release_weekday"])
weekday_dummies = weekday_dummies.drop('Monday', axis=1)
### ADDING ALL DUMMIES AND DATES TOGETHER AND DROPPING CATEGORICAL
release_dates = pd.concat([release_dates, weekday_dummies], axis=1)
release_dates = release_dates.drop(['release_month', 'release_year', 'release_day', 'release_weekday'], axis=1)

######## PLATFORMS ########
#Get Multiple value dummies
platform_dummies = df.platforms.str.get_dummies(sep=";")
## Drop Linux for multicollinearity
platform_dummies = platform_dummies.drop('linux', axis=1)

######## REQUIRED AGE ########
#Getting dummies for age requirement
age_required_dummies = pd.get_dummies(df['required_age'], drop_first=True, prefix='age_')

######## OWNERS ########
## Get owners dummies and drop largest value 0 - 20,000
owners_dummies = pd.get_dummies(df.owners, drop_first=True)

######## ESTIMATED REVENUE ########
owners = df.owners.str.split('-')
list_owners = [list(map(int, x)) for x in owners]
## Mean owners
mean_owners = [np.mean(i) for i in list_owners]
# Est. Revenue
estimated_revenue = mean_owners * df.price
estimated_revenue = estimated_revenue.rename('estimated_revenue')

######## PRICE CATEGORIES ##########
prices_categorical = pd.cut(df.price, [-float('inf'),0,5,10,20,40,100,450], labels=['free','very_cheap', 'cheap', 'avg_price', 'slightly_expensive', 'expensive', 'very_expensive'])
prices_dummies = pd.get_dummies(prices_categorical)

########## CATEGORIES ####################
categories_dummies = df.categories.str.get_dummies(';')
categories_dummies = categories_dummies.loc[:, (categories_dummies).sum() > 50]

####### ADDING COLUMNS ########
new_cols = pd.concat([platform_dummies, age_required_dummies, owners_dummies, estimated_revenue, prices_dummies, categories_dummies], axis=1)

drop_cols = ['genres', 'release_date', 'developer', 'publisher', 'platforms', 'categories', 'steamspy_tags', 'owners']
df = df.drop(drop_cols, axis=1)
# FINAL DATAFRAME
final_df = pd.concat([df.iloc[:,:10], new_cols, release_dates, df.iloc[:,10:]], axis=1)

