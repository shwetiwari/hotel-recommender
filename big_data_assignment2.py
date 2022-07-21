import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from datetime import datetime
import random
import warnings
import math
from sklearn.metrics import ndcg_score
import xgboost as xgb

random.seed(314)
PATH = "training_set_VU_DM.csv"
PATH2 = "test_set_VU_DM.csv"

sns.set(style="ticks", color_codes=True)

np.set_printoptions(threshold=sys.maxsize)  # show full table
np.seterr(divide='ignore', invalid='ignore')  # no warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.width', 10000)
pd.options.display.max_colwidth = 10000

dftrain = pd.read_csv(PATH)
dftrain.drop(['gross_bookings_usd', 'position'], axis=1, inplace=True)

def date_features(df):
    #transforms datetime to year, month and hour
    df[['date', 'time']] = df.date_time.str.split(" ", expand=True)
    df[['year', 'month', 'day']] = df.date.str.split("-", expand=True)
    df[['hour', 'min', 'sec']] = df.time.str.split(":", expand=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['weekday'] = df['date'].dt.dayofweek
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['hour'] = df['hour'].astype(int)
    df = df.drop(['time', 'date', 'day', 'min', 'sec', 'date_time'], axis=1)
    df_date_feat = df

    return df_date_feat


dftrain = date_features(dftrain)
rows = dftrain[dftrain['orig_destination_distance'].isnull()].set_index(['visitor_location_country_id', 'prop_country_id']).index.unique()
dftrain['new_destination_distance'] = dftrain['orig_destination_distance'].copy()
for row in range(len(rows)):

    # test if the same origin destination location is available
    meanDistance = dftrain.loc[(dftrain['prop_country_id'] == rows[row][1]) & (
                dftrain['visitor_location_country_id'] == rows[row][0]), 'orig_destination_distance'].mean()

    # test if the other way around is available otherwise
    if np.isnan(meanDistance):
        meanDistance = dftrain.loc[(dftrain['visitor_location_country_id'] == rows[row][1]) & (
                    dftrain['prop_country_id'] == rows[row][0]), 'orig_destination_distance'].mean()

    # get mean of visitor_location_country_id
    if np.isnan(meanDistance):
        meanDistance = dftrain.loc[
            (dftrain['visitor_location_country_id'] == rows[row][0]), 'orig_destination_distance'].mean()

    if np.isnan(meanDistance):
        meanDistance = dftrain.loc[
            (dftrain['visitor_location_country_id'] == rows[row][1]), 'orig_destination_distance'].mean()

    if np.isnan(meanDistance):
        meanDistance = dftrain.loc[
            (dftrain['prop_country_id'] == rows[row][1]), 'orig_destination_distance'].mean()

    if np.isnan(meanDistance):
        meanDistance = dftrain.loc[
            (dftrain['prop_country_id'] == rows[row][0]), 'orig_destination_distance'].mean()

    if np.isnan(meanDistance):
        meanDistance = dftrain['orig_destination_distance'].mean()

    if np.isnan(meanDistance):
        print("what?")

    dftrain['new_destination_distance'] = np.where(
        ((dftrain['new_destination_distance'].isnull()) & (dftrain['visitor_location_country_id'] == rows[row][0]) & (
                    dftrain['prop_country_id'] == rows[row][1])), meanDistance, dftrain['new_destination_distance'])

dftrain['orig_destination_distance'] = dftrain['new_destination_distance'].copy()
dftrain = dftrain.drop(['new_destination_distance'], axis=1)

# add column whether or not data is available
# for customers
dftrain['returning_customer'] = np.where(
    (dftrain['visitor_hist_starrating'].isnull()) & (dftrain['visitor_hist_adr_usd'].isnull()), 0, 1)

dftrain['visitor_hist_starrating'] = np.where(
    dftrain['visitor_hist_starrating'].isnull(), 0, dftrain['visitor_hist_starrating'])

dftrain['visitor_hist_adr_usd'] = np.where(
    dftrain['visitor_hist_adr_usd'].isnull(), 0, dftrain['visitor_hist_adr_usd'])

# for hotels
dftrain['srch_query_affinity_score'] = np.exp(np.array(dftrain['srch_query_affinity_score']))
dftrain['available_srch_query_affinity_score'] = np.where(dftrain['srch_query_affinity_score'].isnull(), 0, 1)
dftrain['srch_query_affinity_score'] = np.where(
    dftrain['srch_query_affinity_score'].isnull(), 0, dftrain['srch_query_affinity_score'])

dftrain['available_prop_location_score2'] = np.where(dftrain['prop_location_score2'].isnull(), 0, 1)
dftrain['prop_location_score2'] = np.where(dftrain['prop_location_score2'].isnull(), 0, dftrain['prop_location_score2'])

dftrain['available_prop_review_score'] = np.where(
    (dftrain['prop_review_score'].isnull()) | (dftrain['prop_review_score'] == 0), 0, 1)
dftrain['prop_review_score'] = np.where(dftrain['prop_review_score'].isnull(), 0, dftrain['prop_review_score'])

dftrain['available_prop_starrating'] = np.where(dftrain['prop_starrating'] == 0, 0, 1)

# replace competitive numbers by:
# 0: No competitors
# 1: Expedia cheaper then all other competitors
# 2: Hotels can be found somewhere else at least the same price, but not lower
# 3: Expedia is more expensive than at least 1 competitor
comp_invs = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']
comp_rates = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate',
              'comp8_rate']
comp_rate_percs = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff',
                   'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff',
                   'comp7_rate_percent_diff', 'comp8_rate_percent_diff']

dftrain[comp_invs] = dftrain[comp_invs].fillna(1)
dftrain[comp_rates] = dftrain[comp_rates].fillna(0)
dftrain[comp_invs] = 1 - dftrain[comp_invs]
dftrain[comp_invs] = np.where(dftrain[comp_invs] > 1, 1, dftrain[comp_invs])
dftrain[comp_rates] = -1*dftrain[comp_rates]
dftrain[comp_rates] = 10 ** (dftrain[comp_rates] + 1)
dftrain['competitive'] = (dftrain['comp1_inv'] * dftrain['comp1_rate']) + \
                         (dftrain['comp2_inv'] * dftrain['comp2_rate']) + \
                         (dftrain['comp3_inv'] * dftrain['comp3_rate']) + \
                         (dftrain['comp4_inv'] * dftrain['comp4_rate']) + \
                         (dftrain['comp5_inv'] * dftrain['comp5_rate']) + \
                         (dftrain['comp6_inv'] * dftrain['comp6_rate']) + \
                         (dftrain['comp7_inv'] * dftrain['comp7_rate']) + \
                         (dftrain['comp8_inv'] * dftrain['comp8_rate'])

dftrain['competitive'] = np.where((dftrain['competitive'] < 10) & (dftrain['competitive'] > 0), 1, dftrain['competitive'])
dftrain['competitive'] = np.where((dftrain['competitive'] < 100) & (dftrain['competitive'] >= 10), 2, dftrain['competitive'])
dftrain['competitive'] = np.where(dftrain['competitive'] >= 100, 3, dftrain['competitive'])
dftrain.drop(comp_invs + comp_rates + comp_rate_percs, axis=1, inplace=True)

# add new clustering method
dftrain['unique'] = dftrain['visitor_location_country_id'].astype(str) + ',' + \
                    dftrain['srch_adults_count'].astype(str) + ',' + dftrain['srch_children_count'].astype(str) + \
                    ',' + dftrain['site_id'].astype(str)
dftrain['unique'] = dftrain['unique'].astype('category')
dftrain['unique'] = dftrain['unique'].cat.codes
dftrain['unique'] = dftrain['unique'].astype('int64')

# log of price_usd and differences
dftrain['price_usd'] = np.where(dftrain['price_usd'] == 0, 1, dftrain['price_usd'])
dftrain['starrating_diff'] = np.abs(dftrain['visitor_hist_starrating'] - dftrain['prop_starrating'])
dftrain['usd_diff'] = np.abs(dftrain['visitor_hist_adr_usd'] - dftrain['price_usd'])
dftrain['usd_diff'] = np.where(dftrain['usd_diff'] == 0, 1, dftrain['usd_diff'])
dftrain['log_price_usd'] = np.log(dftrain['price_usd'])
dftrain['log_usd_diff'] = np.log(dftrain['usd_diff'])

# normalize in groups
dftrain['price_stand_by_srch_id'] = dftrain.groupby('srch_id')['log_price_usd'].transform(lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['price_stand_by_destination_id'] = dftrain.groupby('srch_destination_id')['log_price_usd'].transform(
    lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['price_stand_by_prop_id'] = dftrain.groupby('prop_id')['log_price_usd'].transform(
    lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['price_stand_by_month'] = dftrain.groupby('month')['log_price_usd'].transform(lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['price_stand_by_unique'] = dftrain.groupby('unique')['log_price_usd'].transform(
    lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['log_price_usd'] = dftrain['log_price_usd'].transform(lambda x: (x - x.mean()) / x.std()).fillna(0)

dftrain['review_stand_by_srch_id'] = dftrain.groupby('srch_id')['prop_review_score'].transform(
    lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['review_stand_by_destination_id'] = dftrain.groupby('srch_destination_id')['prop_review_score'].transform(
    lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['review_stand_by_month'] = dftrain.groupby('month')['prop_review_score'].transform(
    lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['review_stand_by_unique'] = dftrain.groupby('unique')['prop_review_score'].transform(
    lambda x: (x - x.mean()) / x.std()).fillna(0)
dftrain['prop_review_score'] = dftrain['prop_review_score'].transform(lambda x: (x - x.mean()) / x.std()).fillna(0)


dftrain.to_csv('dataprep.csv')

dftrain = pd.read_csv('dataprep.csv', index_col=0)
print(dftrain.min())
print(dftrain.max())
print(dftrain.head())
print(dftrain[pd.isnull(dftrain).any(axis=1)])

def groupsize(df):
    srch_value = df.srch_id.value_counts()
    df_srch_count = pd.DataFrame([srch_value]).T.sort_index()

    return df_srch_count.srch_id

params = {'learning_rate': 0.3, 'gamma': 0, 'min_child_weight': 1,
           'max_depth': 7, 'n_estimators': 100, 'subsample': 1}

x_train = dftrain.drop(['click_bool', 'booking_bool'], axis=1)
y_train = np.array(dftrain['click_bool']) + 4*np.array(dftrain['booking_bool'])
x_test = x_train[4500022:4958347].copy()
y_test = y_train[4500022:4958347].copy()
x_val = x_train[4000003:4500022].copy()
y_val = y_train[4000003:4500022].copy()
x_train = x_train[0:4000003]
y_train = y_train[0:4000003]
print(len(params))

xgb_rank = xgb.XGBRanker(**params)
resultTest = x_test[['srch_id', 'prop_id']].copy()
resultTest['click_bool'] = np.array(dftrain.loc[4500022:4958347, 'click_bool']) + 4*np.array(dftrain.loc[4500022:4958347, 'booking_bool'])

xgb_rank.fit(x_train.drop('srch_id', axis=1), y_train, groupsize(x_train), eval_set=[(x_val.drop('srch_id', axis=1), y_val)],
              eval_group=[groupsize(x_val)], eval_metric='ndcg@5')
evals_result = xgb_rank.evals_result
print(evals_result['eval_0']['ndcg@5'][-1])
preds = xgb_rank.predict(x_test.drop('srch_id', axis=1))[0:458325]

print("Created and trained model")

resultTest["prediction"] = np.array(preds)

resultTest = resultTest.set_index('srch_id')

xgb.plot_importance(xgb_rank)
plt.show()


def dcg_equation(data):
    length = data.shape[0]
    realSortedData = data.sort_values(by=['click_bool'], ascending=False)
    realResult = 0
    predResult = 0
    for index in range(0, min(length, 5)):
        predResult += data.iloc[index][['click_bool']] / math.log2(index + 2)
        realResult += realSortedData.iloc[index][['click_bool']] / math.log2(index + 2)
    return predResult/realResult


uniqueTestIds = resultTest.index.unique()
accuracy = np.zeros(len(uniqueTestIds))
for i in range(len(uniqueTestIds)):
    customerData = resultTest.loc[uniqueTestIds[i]]  # only data from 1 search id
    customerData = customerData.sort_values(by=['prediction'], ascending=False)  # sort on prediction
    accuracy[i] = dcg_equation(customerData)

print(accuracy.mean())