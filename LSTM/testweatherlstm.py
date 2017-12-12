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
from flask import request
from flask import Flask
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
from flask import jsonify
import urllib
import decimal
app = Flask(__name__)
import heapq
import simplejson as json


def return_elevation_distance(location1, location2):
    # p_point_1 = get_polar_points(location1)
    # p_point_2 = get_polar_points(location2)
    # x_1,y_1,z_1,x_2,y_2,z_2 = p_point_1[0], p_point_1[1], p_point_1[2], p_point_2[0], p_point_2[1], p_point_2[2]
    # dist = math.sqrt((x_2-x_1)**2 + (y_2-y_1)**2 + (z_2-z_1)**2)
    # print location1,"hhhhhh"
    return vincenty(location2, location1).meters



@app.route('/getweather', methods=['GET'])
def getInputFromURL():
    # D = decimal.Decimal
    lattitude = request.args.get('lattitude')
    longitude = request.args.get('longitude')
    lattitude = str(lattitude.encode('ascii','ignore'))
    longitude = str(longitude.encode('ascii','ignore'))
    input_date_url = request.args.get("date") 
    input_date_url = str(input_date_url.encode('ascii','ignore')).strip('"\'')
    input_date_url = urllib.unquote(input_date_url).decode('utf8') 
    # print input_date_url,1
    
    # print input_date_url,1
    # input_date_url = input_date_url[0:-1]
    # print input_date_url,2
    # input_date_url = input_date_url.split("-")
    # print input_date_url,3
    # input_date_url = [float(x) for x in input_date_url]
    # input_date_url = [int(x) for x in input_date_url]
    # print input_date_url,4
    # input_date_url = [str(x) for x in input_date_url]
    # input_date_url = '-'.join(input_date_url)
    print input_date_url,5
    temp1= [x for x in lattitude if x.isdigit()]
    # temp1_1, temp1_2 = temp1.split(".")
    # temp1 = str(temp1_1 + "." + temp1_2)
    temp2 = "".join([y for y in longitude])
    # temp2_1, temp2_2 = temp2.split(".")
    # temp2 = str(temp2_1 + "." + temp2_2)
    pointlat = lattitude.index('.')
    if '-' in lattitude: 
        temp1 ='-'+ ''.join(temp1[:pointlat-2])+'.'+''.join(temp1[pointlat-2:])
    else:
        temp1 =''.join(temp1[:pointlat-1])+'.'+''.join(temp1[pointlat-1:])
    pointlong = longitude.index('.')

    temp2= [x for x in longitude if x.isdigit()]
    if '-' in longitude: 
        # print "minus"
        # print "first char latt", lattitude[0]
        temp2 ='-'+ ''.join(temp2[:pointlong-2])+'.'+''.join(temp2[pointlong-2:])
    else:
        # print "plus"
        # print "first char long", longitude[0]
        temp2 = ''.join(temp2[:pointlong-1])+'.'+''.join(temp2[pointlong-1:])
    # print D(temp1), D(temp2)
    # print "lat index, lon index", pointlat, pointlong
    # print "2lattitude", temp1, type(temp1), float(temp1)
    # print "2longitude", temp2, type(temp2), float(temp2)


    #location = (int(lattitude.strip()),int(longitude.strip()))
    location = (temp1, temp2)
    print "date", input_date_url
    ip_date = datetime.datetime.strptime(input_date_url, "%Y-%m-%d")
    distance_array=[]
    station_ids = station_id_lattitude.keys()
    for idx,k in enumerate(station_ids):
        distance_array.append((idx,return_elevation_distance(station_id_lattitude[k],location)))
    distance_array.sort(key = lambda x: x[1])
    distance_array = distance_array[:10]
    nearest_stations = [station_ids[distance_array[i][0]] for i in range(10)]
    print nearest_stations
    # 10_nearest = [heapq.heappop(d_arr) for i in range(10)]
    # 10_nearest_idx = [10_nearest[i] for i in range(10)]
    # min_distance = min(distance_array)
    # min_dist_idx = distance_array.index(min_distance)
    # nearest_station = station_ids[min_dist_idx]

    testreframed=createset("11test.csv", nearest_stations, ip_date)
    test = testreframed.values
    print "test", test
    n_obs = n_hours * n_features
    test_X, test_y = test[:, :n_obs], test[:, -n_features:-6]

    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

    # design network


    # make a prediction

    model = load_model('pickle/lstm_model.h5')
    print "test_X", test_X
    yhat = model.predict(test_X)
    fn = "pickle/minmaxscaler.pkl"

    scaler=pickle.load(open(fn, 'rb'))
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    # invert scaling for forecast
    inv_yhat =yhat #np.concatenate(( test_X[:, :3],yhat), axis=1)
    print "inv_yhat", inv_yhat
    inv_yhat = scaler.inverse_transform(inv_yhat)#[:,3:]



    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 9))
    inv_y = test_y #np.concatenate(( test_X[:, :3],test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)#[:,3:]


    aarmse = sqrt(mean_squared_error(inv_y[:,0], inv_yhat[:,0]))
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    print('Test RMSE: %.3f' % rmse)

    ll=['temp', 'dewp', 'stp', 'visib', 'wdsp', 'mxpsd', 'max', 'min','prcp']
    #print "",randrow,"randrow"
    print "diff: ",zip(ll, mean_squared_error(inv_y, inv_yhat,multioutput='raw_values'))
    print "orig:" ,dict(zip(ll, map(lambda x: round(x, 2),inv_y[0])))
    orig = dict(zip(ll, map(lambda x: round(x, 2),inv_y[0])))
    print "pred:" ,dict(zip(ll, map(lambda x: round(x, 2),inv_yhat[0])))
    pred = dict(zip(ll, map(lambda x: round(x, 2),inv_yhat[0])))


    print('\n\nTest aaaRMSE: %.3f' % aarmse)
    ll=['fog','rain_drizzle','snow_ice_pellets','hail','thunder','tornado_funnel_cloud']
    print "orig:" ,zip(ll,map(int,test[0, -6:]))
    orig_labels = dict(zip(ll,map(int,test[0, -6:])))
    fn = "pickle/scaler.pkl"
    scaler = pickle.load(open(fn, 'rb'))
    xtest = scaler.transform(inv_yhat)
    fn = "pickle/labelpred.pkl"
    clf = pickle.load(open(fn, 'rb'))
    y_pred = clf.predict(xtest)
    pred_labels= dict(zip(ll,y_pred[0]))

    print "pred label:" ,zip(ll,y_pred[0])
    return_dict = {}
    # return_dict["orig_values"] = ','.join(map(str,inv_y.tolist()[0]))
    # return_dict["pred_values"] = ','.join(map(str,inv_yhat.tolist()[0]))
    # return_dict["orig_labels"] = ','.join(map(str,y_pred.tolist()[0]))
    # return_dict["pred_labels"] = ','.join(map(str,y_pred.tolist()[0]))
    return_dict["orig_values"] = orig
    return_dict["pred_values"] = pred
    return_dict["orig_labels"] = orig_labels
    return_dict["pred_labels"] = pred_labels
    print return_dict
    return json.dumps(return_dict)
# convert series to supervised learning
def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
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
    print cols
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
    # print station_id_wban
    datasetDD = read_csv(fname, header=0, index_col=0,parse_dates=['date'])
    end_day = ip_date - datetime.timedelta(days=4)
    for st in station_id_wban:
        # print st
        dataset=datasetDD.loc[((datasetDD['date']>=end_day) & (datasetDD['date']<=ip_date) & (datasetDD["prim_key"]==st) )]
        if dataset.shape[0]==5:
            break



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
    # dfsc = dfsc.loc[dfsc[15] == station_id_wban ]
    print dfsc
    # start_day = ip_date - datetime.timedelta(days=1)
    # end_day = ip_date - datetime.timedelta(days=4)
    # print start_day, end_day
    # print start_day, end_day
    # dfsc[16] = map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), dfsc[16])


    # dfsc = dfsc.loc[((dfsc[16] >= end_day) & (dfsc[16] <= ip_date)) & (dfsc[15] == station_id_wban)]
    print "27 should come here",dfsc
    
    # specify the number of lag hours

    # frame as supervised learning
    reframed = []
    cc = 0
    for region, df_region in dfsc.groupby(15):#15=primkey
        print df_region
        df_region = df_region.drop(15, axis=1)
        print df_region
        df_region.index = df_region[16]# 16 = date
        print df_region
        df_region = df_region.drop(16, axis=1)
        print df_region
        df_region.sort_index(inplace=True)
        print df_region

        cc += df_region.shape[0]
        reframed.append(series_to_supervised(df_region, n_hours, 1))

    reframed = pd.concat(reframed, ignore_index=True)
    return reframed

# split into train and test sets




if __name__ == "__main__":
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
    # input_location = (33.7789949 -84.413332)
    # ip_date = '2017-02-09'
    station_id = df["prim_key"]
    for index,id_stn in enumerate(station_id):
        station_id_lattitude[id_stn] = lat_long_elev[index]
    
    app.run(debug=True)

