import os
import joblib
import pandas as pd
import numpy as np

from multiprocessing.pool import ThreadPool
from pykrige.uk import UniversalKriging

from train_models import prepare_df

from train_models import WINDOW_SIZE
from train_models import DF_COW
from train_models import COW_WEATHER_COLS
from train_models import COW_WEATHER_MODEL
from constants import COW_AGG
from constants import COW_RESULTS

from constants import LOCATION_MAPPING_AT
from constants import SIZE_X
from constants import SIZE_Y
from constants import MIN_LON
from constants import MAX_LON
from constants import MIN_LAT
from constants import MAX_LAT
from constants import MAP_KRIGED
from constants import KM_LON
from constants import KM_LAT
from constants import SEED

np.random.seed(SEED)


def relative_difference(predicted, map_value):
    array = 2 / (1 + np.exp(abs(predicted - map_value)))
    return array


def l2norm_km(a, b, size_x=SIZE_X, size_y=SIZE_Y, km_lon=KM_LON, km_lat=KM_LAT):
    a = np.array(a)
    b = np.array(b)
    scale_x = km_lon / size_x
    scale_y = km_lat / size_y
    return np.sqrt((a[0] * scale_x - b[0] * scale_x)**2 + (a[1] * scale_y - b[1] * scale_y)**2)


def kriging(df, field='Temperature'):
    mapping = LOCATION_MAPPING_AT

    interpolated_map = []
    grid_x = np.arange(MIN_LON, MAX_LON, (MAX_LON - MIN_LON) / SIZE_X)
    grid_y = np.arange(MIN_LAT, MAX_LAT, (MAX_LAT - MIN_LAT) / SIZE_Y)

    high = df[field].max()
    low = df[field].min()
    v_params = {'sill': high, 'range': high - low, 'nugget': low, 'slope': high - low}

    old_month = 0

    groups = df.groupby('datetime')
    for idx in groups.indices:
        if idx.month > old_month:
            old_month = idx.month
            print('Kriging', idx)

        tmp_df = df.loc[df['datetime'] == idx]
        tmp_df = tmp_df.groupby('postal_code').mean()
        tmp_df = tmp_df.reset_index()

        points_x = [mapping['lon'].loc[mapping['zip'] == station].to_numpy().item() for station in tmp_df['postal_code'].to_list()]
        points_y = [mapping['lat'].loc[mapping['zip'] == station].to_numpy().item() for station in tmp_df['postal_code'].to_list()]
        real_values = tmp_df[field].to_numpy().squeeze()

        uk = UniversalKriging(points_x, points_y, real_values, variogram_model='linear', variogram_parameters=v_params)
        z, ss = uk.execute('grid', grid_x, grid_y)
        interpolated_map.append(z.transpose())

    interpolated_map = np.array(interpolated_map)
    return interpolated_map


def localize(df, window_size=WINDOW_SIZE):
    def localization_process(args):
        days, duration, run, start, tmp_df, temperature_maps = args

        columns = ['{}_{}'.format(c,i) for i in range(WINDOW_SIZE) for c in COW_WEATHER_COLS[0]]
        unknown_sensor_values = tmp_df[start:start+duration][columns]

        unknown_station_weather_prediction = weather_model.predict(unknown_sensor_values)

        # localisation
        prior = np.zeros((temperature_maps.shape[1], temperature_maps.shape[2]))
        prior[:, :] = 1 / temperature_maps[0, :, :].size

        for tick in range(duration):
            corr = relative_difference(unknown_station_weather_prediction[tick], temperature_maps[tick + start, :, :])
            tmp = np.multiply(corr, prior)
            prior = tmp / np.sum(tmp)

        x, y = np.unravel_index(prior.argmax(), prior.shape)
        result = l2norm_km((real_x, real_y), (x, y))
        return {'postal_code': unknown_zip, 'animal_id': unknown_cow, 'real_x': real_x,
                'real_y': real_y, 'duration': days, 'predicted_x': x, 'predicted_y': y,
                'run': run, 'start': start, 'result': result}

    weather_model = joblib.load(COW_WEATHER_MODEL)
    mapping = LOCATION_MAPPING_AT

    df = prepare_df(df, 'animal_id', COW_WEATHER_COLS[0])
    df_results = pd.DataFrame()

    if not os.path.exists(MAP_KRIGED):
        temperature_maps = kriging(df)
        np.save(MAP_KRIGED, temperature_maps)
    else:
        temperature_maps = np.load(MAP_KRIGED)

    durations = {  7:   7 * COW_AGG,
                  31:  31 * COW_AGG,
                  90:  90 * COW_AGG,
                 180: 180 * COW_AGG,
                 270: 270 * COW_AGG,
                 365: 365 * COW_AGG - window_size}

    unknown_cows = np.random.choice(df['animal_id'].unique(), 25, replace=False)

    for unknown_cow in unknown_cows:
        print('Running detection for', unknown_cow)

        unknown_zip = df.loc[df['animal_id'] == unknown_cow]['postal_code'].unique().item()
        lon = mapping.loc[mapping['zip'] == unknown_zip]['lon'].squeeze()
        lat = mapping.loc[mapping['zip'] == unknown_zip]['lat'].squeeze()
        grid_x = np.arange(MIN_LON, MAX_LON, (MAX_LON - MIN_LON) / SIZE_X)
        grid_y = np.arange(MIN_LAT, MAX_LAT, (MAX_LAT - MIN_LAT) / SIZE_Y)
        real_x = np.abs(grid_x - lon).argmin()
        real_y = np.abs(grid_y - lat).argmin()

        tmp_df = df.loc[df['animal_id'] == unknown_cow]

        arguments = []
        for days, duration in durations.items():
            for run in range(100):
                if days != 365:
                    start = np.random.randint(0, temperature_maps.shape[0] - duration - window_size)
                else:
                    # 365 days always have the same start, we only have to run it once
                    if arguments[-1][0] == 270 and arguments[-1][2] == 99:
                        start = 0
                    else:
                        break

                arguments.append((days, duration, run, start, tmp_df, temperature_maps))

        p = ThreadPool()
        results = p.map(localization_process, arguments)
        df_results = df_results.append(pd.DataFrame(results), ignore_index=True)
        df_results.to_csv(COW_RESULTS, index=False)


if __name__ == '__main__':
    df_cow = pd.read_csv(DF_COW)
    df_cow['datetime'] = pd.to_datetime(df_cow['datetime'], format='%Y-%m-%d %H:%M:%S')

    df_cow = df_cow.groupby(['animal_id', pd.Grouper(key='datetime', freq='4h')]).mean()
    df_cow = df_cow.reset_index()
    max_rows = df_cow.groupby('animal_id').count()[COW_WEATHER_COLS[1]].max().squeeze()
    df_cow = df_cow.groupby('animal_id').filter(lambda x: x.shape[0] >= max_rows)
    df_cow = df_cow.reset_index()

    localize(df_cow)
