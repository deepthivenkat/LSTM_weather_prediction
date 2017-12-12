from math import sqrt
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import cPickle as pickle
from keras.models import load_model
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import math
from geopy.distance import vincenty
import datetime
from sklearn.metrics import r2_score
from sklearn.svm import SVR

def return_elevation_distance(location1, location2):
    # p_point_1 = get_polar_points(location1)
    # p_point_2 = get_polar_points(location2)
    # x_1,y_1,z_1,x_2,y_2,z_2 = p_point_1[0], p_point_1[1], p_point_1[2], p_point_2[0], p_point_2[1], p_point_2[2]
    # dist = math.sqrt((x_2-x_1)**2 + (y_2-y_1)**2 + (z_2-z_1)**2)
    # print location1,"hhhhhh"
    return vincenty(location2, location1).meters


# convert series to supervised learning
def series_to_supervised(df, ip_date=None,n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(df) is list else df.shape[1]
    # df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset


n_hours = 4
n_features = 15


def createset(fname,station_id_wban, ip_date):
    dataset = read_csv(fname, header=0, index_col=0)
    values = dataset.values
    values[:, :-2] = values[:, :-2].astype('float32')
    # normalize features
    scaled = values.copy()
    kk=scaled[:, :-8]

    fn = "pickle/minmaxscaler.pkl"
    if "train" in fname:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled[:, :-8] = scaler.fit_transform(kk)

        pickle.dump(scaler, open(fn, 'wb'))
    else:
        scaler=pickle.load(open(fn, 'rb'))
        scaled[:, :-8] = scaler.transform(kk)

    dfsc = DataFrame(scaled)
    dfsc = dfsc.loc[dfsc[15] == station_id_wban ]
    start_day = input_date - datetime.timedelta(days=1)
    end_day = input_date - datetime.timedelta(days=4)
    # print start_day, end_day
    dfsc[16] = map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), dfsc[16])

    dfsc = dfsc.loc[((dfsc[16] >= end_day) & (dfsc[16] <= input_date))]
    # print dfsc
    # dfsc = dfsc.loc[(dfsc[16]==ip_date)]
    # dfsc = dfsc[dfsc[15]==station_id_wban and dfsc[16]==ip_date]
    # print dfsc
    # specify the number of lag hours

    # frame as supervised learning
    reframed = []
    cc = 0
    for region, df_region in dfsc.groupby(15):#15=primkey
        df_region = df_region.drop(15, axis=1)
        df_region.index = df_region[16]# 16 = date
        df_region = df_region.drop(16, axis=1)
        df_region.sort_index(inplace=True)

        cc += df_region.shape[0]
        reframed.append(series_to_supervised(df_region, ip_date,n_hours, 1))
        # print reframed

    reframed = pd.concat(reframed, ignore_index=True)
    return reframed

# split into train and test sets
data = defaultdict(tuple)
station_id_lattitude = defaultdict(tuple)
# df = pd.DataFrame()
files = ["2017fff.csv","2016fff.csv","2015fff.csv"]
df_files = []
for idx,f in enumerate(files):
    df_files.append(pd.read_csv(f))

df = pd.concat(df_files, ignore_index=True)
# df.append(pd.read_csv("2015fff.csv"))
print len(df), [len(x) for x in df_files]


    # print df_now
    # df.append(df_now)
# print df
# df = pd.read_csv("2015fff.csv")
lat_long_elev = zip(df["lat"], df["lon"])
df['prim_key'] = df['a_stn'].map(str) + '_'+ df['a_wban'].map(str) 
df['year_month'] = df['year'].astype(str) +'-'+ df['mo'].astype(str) 
df['day_of_the_year'] = df['year_month'].astype(str) +'-'+ df['da'].astype(str) 
dates = []
for x in df['day_of_the_year']:
    dt = datetime.datetime.strptime(x, "%Y-%m-%d")
    dates.append(dt)
df['day_of_the_year'] = pd.Series(dates)
# df['day_of_the_year'] = datetime.datetime.strptime("2015-01-30", "%Y-%m-%d")df['day_of_the_year'].map()
df = df.drop('year',axis=1)
df = df.drop('mo',axis=1)
df = df.drop('a_stn', axis=1)
df = df.drop('a_wban', axis=1)
df = df.drop('lat', axis=1)
df = df.drop('lon', axis=1)
df = df.drop('year_month',axis=1)
df = df.drop('day_of_the_year',axis=1)
df = df.drop('da',axis=1)
df = df.drop('f0_',axis=1)
df['day_of_the_year'] = pd.Series(dates)
input_location = (33.7789949 -84.413332)
input_date = datetime.datetime.strptime('2017-08-28', "%Y-%m-%d")    
station_id = df["prim_key"]
for index,id_stn in enumerate(station_id):
    station_id_lattitude[id_stn] = lat_long_elev[index]
distance_array=[]
station_ids = station_id_lattitude.keys()
for k in station_ids:
    distance_array.append(return_elevation_distance(station_id_lattitude[k],input_location))
min_distance = min(distance_array)
min_dist_idx = distance_array.index(min_distance)
nearest_station = station_ids[min_dist_idx]


testreframed=createset("11test.csv",nearest_station,input_date)
# randrow=random.randint(0,testreframed.shape[0]-1)
# print "randrow",randrow
# test = testreframed.values
# print "test_file",len(test[0])
# test= test[randrow:randrow+1,:]
# print "test data row", test

test = testreframed.values
n_obs = n_hours * n_features
test_X, test_y = test[:, :n_obs], test[:, -n_features:-6]

test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

# design network


# make a prediction

model = load_model('pickle/lstm_model.h5')
yhat = model.predict(test_X)
fn = "pickle/minmaxscaler.pkl"

scaler=pickle.load(open(fn, 'rb'))
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
inv_yhat =yhat #np.concatenate(( test_X[:, :3],yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)#[:,3:]



# invert scaling for actual
test_y = test_y.reshape((len(test_y), 9))
inv_y = test_y #np.concatenate(( test_X[:, :3],test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)#[:,3:]


aarmse = sqrt(mean_squared_error(inv_y[:,0], inv_yhat[:,0]))
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)

ll=['temp', 'dewp', 'stp', 'visib', 'wdsp', 'mxpsd', 'max', 'min','prcp']
# print "",randrow,"randrow"
print "diff: ",zip(ll, mean_squared_error(inv_y, inv_yhat,multioutput='raw_values'))
print "orig:" ,zip(ll, map(lambda x: round(x, 2),inv_y[0]))
print "pred:" ,zip(ll, map(lambda x: round(x, 2),inv_yhat[0]))

# print('\n\nTest aaaRMSE: %.3f' % aarmse)
# ll=['fog','rain_drizzle','snow_ice_pellets','hail','thunder','tornado_funnel_cloud']
# print "orig:" ,zip(ll,map(int,test[0, -6:]))
# fn = "pickle/scaler.pkl"
# scaler = pickle.load(open(fn, 'rb'))
# xtest = scaler.transform(inv_yhat)
# fn = "pickle/labelpred.pkl"
# clf = pickle.load(open(fn, 'rb'))
# y_pred = clf.predict(xtest)

# print "pred label:" ,zip(ll,y_pred[0])



