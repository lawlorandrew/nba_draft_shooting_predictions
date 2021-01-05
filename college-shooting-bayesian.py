from scipy.stats import betabinom, norm, yeojohnson
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
import statsmodels.api as sm
from scipy.stats import skew
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def betabinom_func(params, *args):
    a, b = params[0], params[1]
    k = args[0] # hits
    n = args[1] # at_bats
    return -np.sum(betabinom.logpmf(k, n, a, b))

def solve_a_b(hits, at_bats, max_iter=250):
    result = minimize(betabinom_func, x0=[1, 10],
              args=(hits, at_bats), bounds=((0, None), (0, None)),
              method='L-BFGS-B', options={'disp': True, 'maxiter': max_iter})
    a, b = result.x[0], result.x[1]
    return a, b

# Sanity check your data to ensure hits <= at_bats, at_bats > 0, and both are type int
def estimate_eb(hits, at_bats):
    a, b = solve_a_b(hits, at_bats)
    print(a,b)
    return ((hits+a) / (at_bats+a+b))

def safe_divide(a, b):
  if (b == 0):
    return 0
  else:
    return a/b

raw_df = pd.read_csv('./data/torvik-shooting.csv')
raw_df.loc[raw_df['2PA'].isnull(), '2PA'] = 0
raw_df.loc[raw_df['3PA'].isnull(), '3PA'] = 0
raw_df.loc[raw_df['FTA'].isnull(), 'FTA'] = 0
raw_df.loc[raw_df['Far2'].isnull(), 'Far2'] = 0
raw_df.loc[raw_df['Far2A'].isnull(), 'Far2A'] = 0
raw_df.loc[raw_df['3P'].isnull(), '3P'] = 0
raw_df.loc[raw_df['FT'].isnull(), 'FT'] = 0
raw_df['FGA'] = raw_df['2PA'] + raw_df['3PA']
raw_df['FG'] = raw_df['2P'] + raw_df['3P']
raw_df['3PAR'] = raw_df.apply(lambda row: safe_divide(row['3PA'], row['FGA']), axis=1)
# raw_df = raw_df[raw_df['Season'] > 2009]
raw_df_player_grouped = raw_df.groupby(['Name', 'PlayerID'])
career_df_sums = raw_df_player_grouped.sum().loc[:,['3PA', '3P', 'Far2', 'Far2A', 'FGA', 'FT', 'FTA']]
career_df_min = raw_df_player_grouped['Season'].min()
career_df_max = raw_df_player_grouped['Season'].max()
career_df = career_df_sums.join(career_df_min)
career_df = career_df.join(career_df_max, lsuffix='_Min', rsuffix='_Max')
career_df['career_3p%_eb'] = estimate_eb(career_df['3P'], career_df['3PA'])
career_df['career_ft%_eb'] = estimate_eb(career_df['FT'], career_df['FTA'])
career_df['career_far2%_eb'] = estimate_eb(career_df['Far2'], career_df['Far2A'])
career_df['career_3par_eb'] = estimate_eb(career_df['3PA'], career_df['FGA'])
career_df['career_3p%'] = career_df.apply(lambda row: safe_divide(row['3P'], row['3PA']), axis=1)
career_df['career_ft%'] = career_df.apply(lambda row: safe_divide(row['FT'], row['FTA']), axis=1)
career_df['career_far2%'] = career_df.apply(lambda row: safe_divide(row['Far2'], row['Far2A']), axis=1)
career_df['career_3par'] = career_df.apply(lambda row: safe_divide(row['3PA'], row['FGA']), axis=1)
career_df = career_df.reset_index()
# print(career_df.nlargest(20, 'career_3p%_eb').loc[:,['Name', 'career_3p%_eb']])
raw_df['3p%_eb'] = estimate_eb(raw_df['3P'], raw_df['3PA'])
raw_df['3P%'] = raw_df['3P'] / raw_df['3PA']
raw_df['ft%_eb'] = estimate_eb(raw_df['FT'], raw_df['FTA'])
raw_df['FT%'] = raw_df['FT'] / raw_df['FTA']
raw_df['Far2%'] = raw_df['Far2'] / raw_df['Far2A']
raw_df.loc[raw_df['FT%'].isnull(), 'FT%'] = 0
raw_df.loc[raw_df['3P%'].isnull(), '3P%'] = 0
raw_df.loc[raw_df['Far2%'].isnull(), 'Far2%'] = 0
raw_df['far2p%_eb'] =  estimate_eb(raw_df['Far2'], raw_df['Far2A'])
raw_df['3par_eb'] =  estimate_eb(raw_df['3PA'], raw_df['FGA'])
raw_df['3par'] =  raw_df['3PA'] / raw_df['FGA']
raw_df.loc[raw_df['3par'].isnull(), '3par'] = 0

# raw_df = raw_df[raw_df['Height'].notnull()]
# raw_df = raw_df[raw_df['Height'] != '-']
# raw_df = raw_df[~raw_df['Height'].str.contains("'")]
# raw_df = raw_df[raw_df['Height'].str.contains('5|6|7')]
# raw_df['Height_Split'] = raw_df['Height'].str.split('-')
# raw_df['Feet'] = raw_df['Height_Split'].str[0]
# raw_df['Inches'] = raw_df['Height_Split'].str[1]
# print(raw_df[raw_df['Feet'] == ''].loc[:,['Name', 'Season', 'Height']])
# raw_df['Height_Inches'] = raw_df['Height_Split'].str[0].astype(int)*12 + raw_df['Height_Split'].str[1].astype(int)
# print(raw_df['Height_Inches'].describe())
# print(raw_df['Height_Inches'])
# print(raw_df['Role'].unique())
raw_df = raw_df[raw_df['Class'].notnull()]
raw_df = raw_df[raw_df['Class'] != 'None']
raw_df.loc[raw_df['Role'] == 'Pure PG', 'Role'] = 1
raw_df.loc[raw_df['Role'] == 'Scoring PG', 'Role'] = 1
raw_df.loc[raw_df['Role'] == 'Combo G', 'Role'] = 1
raw_df.loc[raw_df['Role'] == 'Wing G', 'Role'] = 2
raw_df.loc[raw_df['Role'] == 'Wing F', 'Role'] = 2
raw_df.loc[raw_df['Role'] == 'Stretch 4', 'Role'] = 2
raw_df.loc[raw_df['Role'] == 'PF/C', 'Role'] = 3
raw_df.loc[raw_df['Role'] == 'C', 'Role'] = 3
raw_df.loc[raw_df['Class'] == 'Sr', 'Class'] = 4
raw_df.loc[raw_df['Class'] == 'Jr', 'Class'] = 3
raw_df.loc[raw_df['Class'] == 'So', 'Class'] = 2
raw_df.loc[raw_df['Class'] == 'Fr', 'Class'] = 1
raw_df['Role'] = raw_df['Role'].astype('float64')
raw_df['Class'] = raw_df['Class'].astype('float64')

last_year_only = raw_df.sort_values('Season', ascending=False).drop_duplicates(['Name', 'PlayerID'])
last_year_only = pd.merge(last_year_only, career_df.loc[:,['Name','PlayerID','career_3p%_eb','career_far2%_eb','career_3par_eb','career_ft%_eb', 'career_3p%', 'career_ft%', 'career_far2%', 'career_3par']], on=['Name', 'PlayerID'])
first_year_only = raw_df.sort_values('Season', ascending=True).drop_duplicates(['Name', 'PlayerID'])
last_year_only = pd.merge(last_year_only, first_year_only.loc[:,['Name', 'PlayerID', '3P%', 'FT%', '3PAR']], on=['Name', 'PlayerID'], suffixes=('', '_first'))
last_year_only['3P%_delta'] = last_year_only['3P%'] - last_year_only['3P%_first']
last_year_only['FT%_delta'] = last_year_only['FT%'] - last_year_only['FT%_first']
last_year_only['3PAR_delta'] = last_year_only['3PAR'] - last_year_only['3PAR_first']
nba_df = pd.read_csv('./data/draft/blocks.csv')
nba_df = nba_df[nba_df['g'] > 0]
nba_df['NBA_3p%_eb'] = estimate_eb(nba_df['fg3'], nba_df['fg3a'])
# nba_df = nba_df[nba_df['player'] != 'Chris Wright'] # ignoring for now bc of dupes

merged_df = pd.merge(last_year_only, nba_df, left_on=['Name'], right_on=['player'])
# merged_df[merged_df['3PA'].isnull()].loc[:, 'player'].to_csv('unmatched.csv')
merged_df = merged_df[merged_df['fg3_pct'].notnull()]
# print(merged_df['noooo'])
merged_df = merged_df[(merged_df['fg3a'] >= 100) & (merged_df['3PA'] > 0)]
y = merged_df['fg3_pct']
continuous_vs = ['3P%', 'FT%', '3PAR', 'Far2%', 'career_3p%', 'career_ft%', 'career_3par', 'career_far2%']
# continuous_vs = ['Role', '3P%', 'FT%', '3par', 'Far2%']
# continuous_vs = ['Role', '3p%_eb', 'ft%_eb', '3par_eb', 'far2p%_eb']
noncontinuous_vs = []
binned_vs = []
cols = continuous_vs + noncontinuous_vs + binned_vs
X = merged_df.loc[:, cols]

# fig, ax = plt.subplots(1,2)
# ax[0].scatter(merged_df['3p%_eb'], merged_df['NBA_fg3_pct'], s=2, color='black')
# three_point_eb_reg = LinearRegression().fit(merged_df.loc[:,['3p%_eb']], merged_df['NBA_fg3_pct'])
# print(three_point_eb_reg.coef_)
# print(three_point_eb_reg.intercept_)
# print(three_point_eb_reg.score(merged_df.loc[:,['3p%_eb']], merged_df['NBA_fg3_pct']))
# ax[0].plot(merged_df['3p%_eb'], three_point_eb_reg.coef_[0]*merged_df['3p%_eb'] + three_point_eb_reg.intercept_, color='black')
# ax[0].set_xlabel('College Empirical Bayesian 3P%')
# ax[0].set_ylabel('NBA 3P%')
# ax[0].set_title("EB 3P% R^2: {:.2f}".format(three_point_eb_reg.score(merged_df.loc[:,['3p%_eb']], merged_df['NBA_fg3_pct'])))
# ax[1].scatter(merged_df['3P%'], merged_df['NBA_fg3_pct'], s=2, color='black')
# three_point_reg = LinearRegression().fit(merged_df.loc[:,['3P%']], merged_df['NBA_fg3_pct'])
# print(three_point_reg.coef_)
# print(three_point_reg.intercept_)
# print(three_point_reg.score(merged_df.loc[:,['3P%']], merged_df['NBA_fg3_pct']))
# ax[1].plot(merged_df['3P%'], three_point_eb_reg.coef_[0]*merged_df['3P%'] + three_point_eb_reg.intercept_, color='black')
# ax[1].set_xlabel('College 3P%')
# ax[1].set_ylabel('NBA 3P%')
# ax[1].set_title("3P% R^2: {:.2f}".format(three_point_reg.score(merged_df.loc[:,['3P%']], merged_df['NBA_fg3_pct'])))
# plt.savefig('./output/Comparing R^2 for 3P% and EB 3P%')
# plt.close('all')

# fig, ax = plt.subplots()
# scatter = ax.scatter(raw_df['Far2%'], raw_df['far2p%_eb'], c=raw_df['Far2A'], cmap='RdPu', s=2)
# ax.set_xlabel('College Far 2%')
# ax.set_ylabel('College Empirical Bayesian Far 2%')

# # produce a legend with the unique colors from the scatter
# cb = plt.colorbar(scatter)
# cb.set_label('3PA')
# plt.savefig('./output/College EB Far 2%.png')

# early_entrants_df = pd.read_csv('./data/draft/early-entrants.csv')
# seniors_df = pd.read_csv('./data/draft/seniors.csv')
draft_rankings_df = pd.read_csv('./data/draft/draft_rankings.csv')
# seniors = ['Desmond Bane', 'Grant Riller', 'Cassius Winston', 'Payton Pritchard', 'Skylar Mays', 'Markus Howard', 'Sam Merrill', 'Killian Tillie']
# draft_class = raw_df[
#   ((raw_df['Name'].isin(early_entrants_df['Player'])) & (raw_df['Season'] == 2020))
#   # ((raw_df['Name'].isin(seniors)) & (raw_df['Season'] == 2020))
# ]
raw_df_2020 = last_year_only[last_year_only['Season'] == 2020]
draft_class = pd.merge(raw_df_2020, draft_rankings_df, how='inner', left_on='Name', right_on='Name')
print(draft_class.nlargest(10,'3p%_eb').loc[:,['Name', '3p%_eb', '3P', '3PA']])
print(draft_class.nlargest(10,'ft%_eb').loc[:,['Name', 'ft%_eb', 'FT', 'FTA']])
print(draft_class.nlargest(10,'far2p%_eb').loc[:,['Name', 'far2p%_eb', 'Far2', 'Far2A']])
print(draft_class.nlargest(10,'3par_eb').loc[:,['Name', '3par_eb', '3PA', 'FGA']])

draft_class_X = draft_class.loc[:, cols]
merged_df_gte = merged_df[merged_df['fg3a'] >= 100]
y_gte = merged_df_gte['fg3_pct']
X_gte = merged_df_gte.loc[:, cols]
print('3P%')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.25, random_state=42)
# print(X_val.shape)
print(X_test.shape)
print(X_train.shape)
scalers = []
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf_params = {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}
# rf = RandomForestRegressor(random_state=42)
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X_train, y_train)
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [1,5,10],
    'max_features': [2, 3, 'sqrt'],
    'min_samples_leaf': [2],
    'min_samples_split': [3,5,7],
    'n_estimators': [800, 1000, 1200]
}
# Create a based model
rf = RandomForestRegressor(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# rfr = grid_search.best_estimator_
rfr = RandomForestRegressor(bootstrap=True, max_depth=5, max_features=2, min_samples_leaf=2, min_samples_split=3, n_estimators=800, random_state=42)
rfr.fit(X_train, y_train)
draft_class['rfr_nba_3p%'] = rfr.predict(draft_class_X)
print('RANDOM FOREST')
print('R^2')
print(rfr.score(X_test, y_test))
print(rfr.score(X_train, y_train))
print(rfr.score(X_gte, y_gte))
print('MAE')
print(mean_absolute_error(y_test, rfr.predict(X_test)))
print(mean_absolute_error(y_train, rfr.predict(X_train)))
print(mean_absolute_error(y_gte, rfr.predict(X_gte)))
print('MSE')
print(mean_squared_error(y_test, rfr.predict(X_test)))
print(mean_squared_error(y_train, rfr.predict(X_train)))
print(mean_squared_error(y_gte, rfr.predict(X_gte)))
print('RMSE')
print(mean_squared_error(y_test, rfr.predict(X_test), squared=False))
print(mean_squared_error(y_train, rfr.predict(X_train), squared=False))
print(mean_squared_error(y_gte, rfr.predict(X_gte), squared=False))
# for v in continuous_vs:
#   X_train[v], l = yeojohnson(X_train[v])
#   # print(v)
#   # print(l)
#   X_gte[v] = yeojohnson(X_gte[v], l)
#   draft_class_X[v] = yeojohnson(draft_class_X[v], l)
#   X_test[v] = yeojohnson(X_test[v], l)
#   X[v] = yeojohnson(X[v], l)
# for v in binned_vs:
#   X_train[v] = pd.qcut(X_train[v], 10, False)
#   X_gte[v] = pd.qcut(X_gte[v], 10, False)
#   draft_class_X[v] = pd.qcut(draft_class_X[v], 10, False)
#   X_test[v] = pd.qcut(X_test[v], 10, False)
#   X[v] = pd.qcut(X[v], 10, False)
# if (len(continuous_vs) > 0):
#   scaler = preprocessing.StandardScaler().fit(X_train.loc[:,continuous_vs])
#   X_train.loc[:,continuous_vs] = scaler.transform(X_train.loc[:,continuous_vs])
#   X_test.loc[:,continuous_vs] = scaler.transform(X_test.loc[:,continuous_vs])
#   X_gte.loc[:,continuous_vs] = scaler.transform(X_gte.loc[:,continuous_vs])
#   X.loc[:,continuous_vs] = scaler.transform(X.loc[:,continuous_vs])
#   draft_class_X.loc[:,continuous_vs] = scaler.transform(draft_class_X.loc[:,continuous_vs])
# h_reg = HuberRegressor().fit(X_train, y_train)
# print(h_reg.score(X_train, y_train))
# print(h_reg.coef_)
# print(X)
# print(h_reg.score(X_test, y_test))
# pred = h_reg.predict(X_train)
# l_reg = LinearRegression().fit(X_train, y_train)
# print(l_reg.score(X_train, y_train))
# print(l_reg.coef_)
# # print(X)
# print(l_reg.score(X_test, y_test))
# pred = l_reg.predict(X_train)
# diffs = pred - y_train
# # plt.scatter(y_train, diffs)
# plt.scatter(l_reg.predict(X_test), y_test)
# x = np.array([0,1])
# plt.plot(x, x)
# plt.show()

# X_gte = poly.transform(X_gte)

# print('LINEAR')
# print('R^2')
# print(l_reg.score(X_test, y_test))
# print(l_reg.score(X_train, y_train))
# print(l_reg.score(X_gte, y_gte))
# print('MAE')
# print(mean_absolute_error(y_test, l_reg.predict(X_test)))
# print(mean_absolute_error(y_train, l_reg.predict(X_train)))
# print(mean_absolute_error(y_gte, l_reg.predict(X_gte)))
# print('MSE')
# print(mean_squared_error(y_test, l_reg.predict(X_test)))
# print(mean_squared_error(y_train, l_reg.predict(X_train)))
# print(mean_squared_error(y_gte, l_reg.predict(X_gte)))
# print('RMSE')
# print(mean_squared_error(y_test, l_reg.predict(X_test), squared=False))
# print(mean_squared_error(y_train, l_reg.predict(X_train), squared=False))
# print(mean_squared_error(y_gte, l_reg.predict(X_gte), squared=False))
# # print('HUBER')
# # print('R^2')
# # print(h_reg.score(X_test, y_test))
# # print(h_reg.score(X_train, y_train))
# # print(h_reg.score(X_gte, y_gte))
# # print('MAE')
# # print(mean_absolute_error(y_test, h_reg.predict(X_test)))
# # print(mean_absolute_error(y_train, h_reg.predict(X_train)))
# # print(mean_absolute_error(y_gte, h_reg.predict(X_gte)))
# # print('MSE')
# # print(mean_squared_error(y_test, h_reg.predict(X_test)))
# # print(mean_squared_error(y_train, h_reg.predict(X_train)))
# # print(mean_squared_error(y_gte, h_reg.predict(X_gte)))
# # print('RMSE')
# # print(mean_squared_error(y_test, h_reg.predict(X_test), squared=False))
# # print(mean_squared_error(y_train, h_reg.predict(X_train), squared=False))
# # print(mean_squared_error(y_gte, h_reg.predict(X_gte), squared=False))


# # draft_class_X = pd.get_dummies(draft_class_X, columns=['Role'])
# draft_class_X['3pt_interaction'] = draft_class_X['3par_eb'] * draft_class_X['3p%_eb']
# draft_class_X = poly.transform(draft_class_X)
stdev = np.sqrt(sum((rfr.predict(X_test) - y_test)**2) / (len(y_test) - 2))
print('TEST STDEV')
print(stdev)
draft_class['pred_nba_3p%'] = rfr.predict(draft_class_X)
gaussian_val = value = norm.ppf(.95)
draft_class['pred_nba_3p%_low'] = rfr.predict(draft_class_X) - gaussian_val*stdev
draft_class['pred_nba_3p%_high'] = rfr.predict(draft_class_X) + gaussian_val*stdev
merged_df['pred_nba_3p%'] = rfr.predict(X)
merged_df['pred_nba_3p%_low'] = rfr.predict(X) - gaussian_val*stdev
merged_df['pred_nba_3p%_high'] = rfr.predict(X) + gaussian_val*stdev
print(draft_class.nlargest(20, 'pred_nba_3p%').loc[:,['Name', 'School', 'pred_nba_3p%', 'pred_nba_3p%_low', 'pred_nba_3p%_high']])
draft_class.loc[:,['Name', 'School', 'pred_nba_3p%', 'pred_nba_3p%_low', 'pred_nba_3p%_high', 'Draft Ranking'] + cols].to_csv('./data/draft/3p_predictions_rf.csv')
merged_df.loc[:, ['Name', 'pred_nba_3p%', 'fg3_pct', 'Class'] + cols].to_csv('past_predictions.csv')
merged_df['error'] = np.abs((merged_df['fg3_pct'] - merged_df['pred_nba_3p%'])/merged_df['pred_nba_3p%'])
sum_by_class = merged_df.groupby('Class').mean()['error']
print(sum_by_class)
print(merged_df.nsmallest(20, 'pred_nba_3p%').loc[:,['Name', 'career_3p%', 'career_ft%', 'career_3par', 'pred_nba_3p%', 'fg3_pct']])
print(merged_df.nlargest(20, 'pred_nba_3p%').loc[:,['Name', 'career_3p%', 'career_ft%', 'career_3par', 'pred_nba_3p%', 'fg3_pct']])
print(merged_df.nlargest(20, 'fg3_pct').loc[:,['Name', 'fg3_pct', 'pred_nba_3p%', 'pred_nba_3p%_high']])
print(draft_class.nsmallest(20, 'pred_nba_3p%').loc[:,['Name', '3PA', '3P', 'pred_nba_3p%', 'pred_nba_3p%_low', 'pred_nba_3p%_high']])
print(rfr.feature_importances_)
print(draft_class.nlargest(20, 'rfr_nba_3p%').loc[:,['Name', 'School', 'rfr_nba_3p%']])