import pandas as pd
import numpy as np
import joblib
import json


def read_csv_and_expand_json(file_path, columns_to_expand):
    df = pd.read_csv(file_path, parse_dates=['visitStartTime'], date_parser=lambda x: pd.to_datetime(x, unit='s'))

    def expand_json_columns(df, columns_list):
        for column_name in columns_list:
            df[column_name] = df[column_name].apply(json.loads)
            expanded_df = pd.json_normalize(df[column_name])
            df = pd.concat([df, expanded_df], axis=1)
            df.drop(columns=[column_name], inplace=True)
        return df

    df = expand_json_columns(df, columns_to_expand)

    numeric_columns = ['visits', 'hits', 'pageviews', 'bounces', 'newVisits']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df[['bounces', 'newVisits']] = df[['bounces', 'newVisits']].fillna(0)

    if df['pageviews'].isnull().any():
        df['pageviews'] = df['pageviews'].fillna(df['hits'])

    df['isMobile'] = df['isMobile'].map({False: 0, True: 1})
    df[['bounces', 'newVisits']] = df[['bounces', 'newVisits']].astype(int)

    df.set_index('visitStartTime', inplace=True)
    df['DayOfWeek'] = df.index.day_name()
    df['Hour'] = df.index.hour
    return df


def batch_predict(model, X, batch_size):
    num_samples = len(X)
    num_batches = int(np.ceil(num_samples / batch_size))
    predictions = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X.iloc[start_idx:end_idx]
        y_batch = model.predict(X_batch)
        predictions.extend(y_batch)

    return np.array(predictions)


columns_to_expand = ['device', 'totals', 'geoNetwork', 'trafficSource']
test_df = read_csv_and_expand_json(r"E:\project\GA\data\train.csv", columns_to_expand)

selected_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'continent', 'subContinent',
                     'country', 'region', 'metro', 'city', 'networkDomain', 'campaign', 'source', 'medium', 'DayOfWeek']
numeric_features = ['visitNumber', 'hits', 'pageviews', 'bounces', 'newVisits', 'Hour']

X_test = test_df[selected_features + numeric_features]

# Load classification model
class_model = joblib.load(r'E:\project\GA\models\best_classification_model.pkl')
batch_size = 500
revenue_status_pred = batch_predict(class_model, X_test, batch_size)
test_df['RevenueStatus'] = revenue_status_pred

# Load regression model
regression_model = joblib.load(r'E:\project\GA\models\LGBM Regressor_model.pkl')
test_df['PredictedLogRevenue'] = 0
test_df.loc[test_df['RevenueStatus'] == 1, 'PredictedLogRevenue'] = regression_model.predict(
    test_df[test_df['RevenueStatus'] == 1])

# test_df['PredictedLogRevenue'] = np.expm1(test_df['PredictedLogRevenue'])
