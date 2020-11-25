import os
import pandas as pd

SIZE_X = 200
SIZE_Y = 100

MIN_LAT = 46
MAX_LAT = 49
MIN_LON = 9
MAX_LON = 17
MAP_KRIGED = 'map_krigged.npy'
KM_LAT = 111
KM_LON = 113

AUTOML_TIME = 3*60*60
SEED = 42
WINDOW_SIZE = 7
VAL_SIZE = 0.2

DF_COW = 'df_at_cow.csv'
AT_WEATHER_DATA = os.path.abspath('data/weather_from_postal_code/*.csv')
COW_DATA = os.path.abspath('data/cows/*.csv')
COW_WEATHER_COLS = (['temp', 'act', 'temp_without_drink_cycles', 'YTD'], ['Temperature'])
COW_WEATHER_MODEL = 'cow_weather_mdl.pkl'
COW_AGG = 24 // 4
COW_RESULTS = 'cow_results.csv'

FONT = dict(size=18)

LOCATION_MAPPING_AT = pd.DataFrame(columns=['zip', 'lat', 'lon', 'province'], data=[
    [1210, 48.283333, 16.412222, 'Vienna'],
    [2381, 48.15, 16.166667, 'Lower Austria'],
    [2831, 47.660278, 16.158333, 'Lower Austria'],
    [2852, 47.45, 16.2, 'Lower Austria'],
    [2860, 47.5, 16.283333, 'Lower Austria'],
    [3143, 48.158889, 15.687222, 'Lower Austria'],
    [3300, 48.123, 14.87213, 'Lower Austria'],
    [3340, 47.966667, 14.766667, 'Lower Austria'],
    [3610, 48.4, 15.466667, 'Lower Austria'],
    [3684, 48.266667, 15.033333, 'Lower Austria'],
    [3720, 48.55, 15.85, 'Lower Austria'],
    [3911, 48.516667, 15.066667, 'Lower Austria'],
    [3920, 48.566667, 14.95, 'Lower Austria'],
    [4091, 48.533333, 13.65, 'Upper Austria'],
    [4131, 48.445, 13.936111, 'Upper Austria'],
    [4133, 48.465833, 13.881944, 'Upper Austria'],
    [4191, 48.552778, 14.22, 'Upper Austria'],
    [4274, 48.394444, 14, 'Upper Austria'],
    [4283, 48.35, 14.666667, 'Upper Austria'],
    [4451, 48.021667, 14.408889, 'Upper Austria'],
    [4483, 48.15, 14.425556, 'Upper Austria'],
    [4582, 47.665278, 14.340833, 'Upper Austria'],
    [4591, 47.883611, 14.258889, 'Upper Austria'],
    [4680, 48.185556, 13.642222, 'Upper Austria'],
    [4753, 48.263611, 13.573611, 'Upper Austria'],
    [4754, 48.265, 13.523889, 'Upper Austria'],
    [4792, 48.483333, 13.566667, 'Upper Austria'],
    [4793, 48.481944, 13.611111, 'Upper Austria'],
    [4794, 48.433056, 13.65, 'Upper Austria'],
    [4863, 47.950278, 13.583611, 'Upper Austria'],
    [4921, 48.194167, 13.544444, 'Upper Austria'],
    [5211, 48, 13.216667, 'Upper Austria'],
    [5300, 47.85, 13.066667, 'Salzburg'],
    [5301, 47.866944, 13.124167, 'Salzburg'],
    [5311, 47.831667, 13.400556, 'Salzburg'],
    [6070, 47.266667, 11.433333, 'Tyrol'],
    [6092, 47.234167, 11.300833, 'Tyrol'],
    [6112, 47.283333, 11.583333, 'Tyrol'],
    [6162, 47.233333, 11.366667, 'Tyrol'],
    [7400, 47.283333, 16.2, 'Burgenland'],
    [7433, 47.366667, 16.233333, 'Burgenland'],
    [8045, 47.140833, 15.490556, 'Styria'],
    [8063, 47.122778, 15.599167, 'Styria'],
    [8076, 47.013333, 15.556389, 'Styria'],
    [8143, 46.946667, 15.376389, 'Styria'],
    [8160, 47.25, 15.594444, 'Styria'],
    [8162, 47.283333, 15.516667, 'Styria'],
    [8163, 47.285833, 15.478056, 'Styria'],
    [8225, 47.301944, 15.833889, 'Styria'],
    [8232, 47.340278, 15.990833, 'Styria'],
    [8333, 47, 15.933333, 'Styria'],
    [8442, 46.786111, 15.45, 'Styria'],
    [8521, 46.83, 15.384167, 'Styria'],
    [8580, 47.066667, 15.083333, 'Styria'],
    [8691, 47.6775, 15.644167, 'Styria'],
    [8700, 47.381667, 15.097222, 'Styria'],
    [8714, 47.3, 14.93, 'Styria'],
    [8715, 47.25, 14.883333, 'Styria'],
    [8723, 47.25, 14.85, 'Styria'],
    [8733, 47.271111, 14.860556, 'Styria'],
    [8741, 47.130833, 14.739444, 'Styria'],
    [8742, 47.068056, 14.695, 'Styria'],
    [8773, 47.38, 14.9, 'Styria'],
    [8793, 47.426111, 15.006667, 'Styria'],
    [8951, 47.533333, 14.083333, 'Styria'],
    [8952, 47.49, 14.098611, 'Styria'],
    [8962, 47.445556, 13.901111, 'Styria'],
    [8967, 47.4075, 13.766944, 'Styria'],
    [8983, 47.555, 13.929167, 'Styria'],
    [9161, 46.55, 14.3, 'Carinthia'],
    [9411, 46.835917, 14.784686, 'Carinthia'],
    [9470, 46.702134, 14.853646, 'Carinthia'],
    [9560, 46.709603, 14.089366, 'Carinthia'],
    [9900, 46.862523, 12.726034, 'Tyrol'],
    [9912, 46.782273, 12.550642, 'Tyrol'],
    [39025, 46.646976, 10.988418, 'South Tyrol']
])
