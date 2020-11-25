import os
import glob
import joblib
import pandas as pd
import numpy as np

import autosklearn.regression
import sklearn
from sklearn.model_selection import train_test_split
from datetime import datetime

from constants import AUTOML_TIME
from constants import WINDOW_SIZE
from constants import SEED
from constants import VAL_SIZE

from constants import DF_COW
from constants import AT_WEATHER_DATA
from constants import COW_DATA
from constants import COW_WEATHER_COLS
from constants import COW_WEATHER_MODEL


def print_metrics(y_val, y_hat):
    print('R2:', sklearn.metrics.r2_score(y_val, y_hat))
    print('MSE:', sklearn.metrics.mean_squared_error(y_val, y_hat))
    print('MAE:', sklearn.metrics.mean_absolute_error(y_val, y_hat))
    print('MedAE:', sklearn.metrics.median_absolute_error(y_val, y_hat))


def prepare_df(df, key, columns, window_size=WINDOW_SIZE):
    df['consecutive'] = df[key].eq(df[key].shift(window_size - 1))

    for i in range(1, window_size):
        for c in columns:
            df['{}_{}'.format(c, i)] = df[c].shift(-i)
    for c in columns:
        df = df.rename(columns={c: c + '_0'})
    df = df.loc[df['consecutive'] == True]
    df = df.dropna()

    return df


def preprocess_cow_data(path_weather, path_animals):
    df_weather = pd.DataFrame()
    for csv in glob.glob(path_weather):
        with open(csv, 'r') as f:
            lines = f.readlines()
            lat = float(lines[3].strip().split(';')[1])
            lon = float(lines[4].strip().split(';')[1])
            plz = csv.split('/')[-1].split('_')[0]

        tmp_df = pd.read_csv(csv, header=22, sep=';')
        tmp_df['postal_code'] = int(plz)
        tmp_df['lat'] = lat
        tmp_df['lon'] = lon
        tmp_df['datetime'] = pd.date_range(start='2019-01-01 01:00', end='2020-01-01', freq='1H')
        tmp_df = tmp_df.drop(columns=['# Date', 'UT time'])
        df_weather = df_weather.append(tmp_df)

    csv_files = sorted(glob.glob(path_animals))
    meta_fn = csv_files.pop(-1)
    meta_df = pd.read_csv(meta_fn)
    df = pd.DataFrame()

    for idx, csv in enumerate(csv_files):
        if csv == meta_fn:
            continue
        else:
            tmp_df = pd.read_csv(csv, delimiter=',')
            animal_id = csv.split('/')[-1].split('.')[0]
            tmp_df['animal_id'] = animal_id
            plz = meta_df['postal_code'].loc[meta_df['animal_id'] == animal_id].squeeze()
            tmp_df['postal_code'] = plz

            tmp_df['datetime'] = pd.to_datetime(tmp_df['datetime'], format='%Y-%m-%dT%H:%M:%S%z')
            tmp_df['datetime'] = tmp_df['datetime'].apply(lambda x: pd.to_datetime(x).tz_convert(None))
            tmp_df = tmp_df.set_index('datetime')

            tmp_weather = df_weather.loc[df_weather['postal_code'] == plz]
            tmp_weather = tmp_weather.drop(columns=['postal_code'])
            tmp_weather = tmp_weather.set_index('datetime')
            tmp_df = pd.merge(tmp_df, tmp_weather, how='outer', left_index=True, right_index=True)
            for c in df_weather.columns.drop('datetime'):
                if pd.api.types.is_numeric_dtype(tmp_df[c]):
                    tmp_df[c] = tmp_df[c].interpolate(method='pad')

            tmp_df = tmp_df.reset_index()
            tmp_df = tmp_df.dropna()
            tmp_df['YTD'] = tmp_df['datetime'].apply(lambda x: (datetime(x.year, x.month, x.day) - datetime(x.year, 1, 1)).days)
            df = df.append(tmp_df)

            print('Preprocessing Cow Data: {:3d}/{:3d}'.format(idx+1, len(csv_files)))

    df = df.reset_index()
    return df


def train_cow_model(df, window_size=WINDOW_SIZE, val_size=VAL_SIZE, per_cow=False):
    if per_cow:
        cow_weather_model = COW_WEATHER_MODEL.replace('mdl', 'per_cow_mdl')
    else:
        cow_weather_model = COW_WEATHER_MODEL

    columns = set([item for sublist in COW_WEATHER_COLS for item in sublist])
    df = prepare_df(df, 'animal_id', columns, window_size)

    input_params = ['{}_{}'.format(c, i) for i in range(window_size) for c in COW_WEATHER_COLS[0]]
    target = '{}_{}'.format(COW_WEATHER_COLS[1][0], window_size-1)

    if per_cow:
        np.random.seed(SEED)

        cows = df['animal_id'].unique()
        stations_val = np.random.choice(cows, int(len(cows) * val_size), replace=False)
        stations_train = set(cows) - set(stations_val)

        query_train = ' | '.join(['animal_id == "{}"'.format(s) for s in stations_train])
        query_val = ' | '.join(['animal_id == "{}"'.format(s) for s in stations_val])

        X_train = df.query(query_train)[input_params]
        y_train = df.query(query_train)[target]
        X_val = df.query(query_val)[input_params]
        y_val = df.query(query_val)[target]
    else:
        X_train, X_val, y_train, y_val = train_test_split(df[input_params], df[target], test_size=val_size, random_state=SEED)

    automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=AUTOML_TIME, memory_limit=16*1024, seed=SEED)
    mdl = automl.fit(X_train, y_train)
    joblib.dump(mdl, cow_weather_model)

    y_hat = mdl.predict(X_val)
    print('#######################################################')
    print(cow_weather_model)
    print_metrics(y_val, y_hat)
    print('#######################################################')


if __name__ == '__main__':
    if not os.path.exists(DF_COW):
        df_cow = preprocess_cow_data(AT_WEATHER_DATA, COW_DATA)
        df_cow.to_csv(DF_COW, index=False)
    else:
        df_cow = pd.read_csv(DF_COW)
        df_cow['datetime'] = pd.to_datetime(df_cow['datetime'], format='%Y-%m-%d %H:%M:%S')

    df_cow = df_cow.groupby(['animal_id', pd.Grouper(key='datetime', freq='4h')]).mean()
    df_cow = df_cow.reset_index()

    train_cow_model(df_cow)
