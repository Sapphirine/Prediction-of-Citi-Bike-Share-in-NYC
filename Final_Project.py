
# Importing libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from termcolor import colored
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

Stations = pd.read_csv('stations.csv', low_memory=False)

OD_2016_123 = pd.read_csv('2016_123_trip.csv', low_memory=False)
OD_2016_04 = pd.read_csv('201604_trip.csv', low_memory=False)
OD_2016_05 = pd.read_csv('201605_trip.csv', low_memory=False)

# Importing the datasets
OD_2017 = pd.read_csv('201701.csv', low_memory=False)

OD_2016_06 = pd.read_csv('201606_trip.csv',usecols=['starttime','start station id','stoptime','end station id','tripduration'],low_memory=False)
OD_2016_06.rename(columns = {"tripduration": "duration_sec", "starttime": "start_date", "stoptime": "end_date","start station id":"start_station_code","end station id":"end_station_code"}, inplace=True)
OD_2016_07 = pd.read_csv('201607_trip.csv',usecols=['starttime','start station id','stoptime','end station id','tripduration'], low_memory=False)
OD_2016_07.rename(columns = {"tripduration": "duration_sec", "starttime": "start_date", "stoptime": "end_date","start station id":"start_station_code","end station id":"end_station_code"}, inplace=True)

OD_2016_08 = pd.read_csv('201608_trip.csv',usecols=['starttime','start station id','stoptime','end station id','tripduration'], low_memory=False)
OD_2016_08.rename(columns = {"tripduration": "duration_sec", "starttime": "start_date", "stoptime": "end_date","start station id":"start_station_code","end station id":"end_station_code"}, inplace=True)

OD_2016_09 = pd.read_csv('201609_trip.csv',usecols=['starttime','start station id','stoptime','end station id','tripduration'], low_memory=False)
OD_2016_09.rename(columns = {"tripduration": "duration_sec", "starttime": "start_date", "stoptime": "end_date","start station id":"start_station_code","end station id":"end_station_code"}, inplace=True)

OD_2016_10 = pd.read_csv('201610_trip.csv',usecols=['Start Time','Start Station ID','Stop Time','End Station ID','Trip Duration'], low_memory=False)
OD_2016_10.rename(columns = {"Trip Duration": "duration_sec", "Start Time": "start_date", "Stop Time": "end_date","Start Station ID":"start_station_code","End Station ID":"end_station_code"}, inplace=True)

OD_2016_11 = pd.read_csv('201611_trip.csv',usecols=['Start Time','Start Station ID','Stop Time','End Station ID','Trip Duration'], low_memory=False)
OD_2016_11.rename(columns = {"Trip Duration": "duration_sec", "Start Time": "start_date", "Stop Time": "end_date","Start Station ID":"start_station_code","End Station ID":"end_station_code"}, inplace=True)

OD_2016_12 = pd.read_csv('201612_trip.csv',usecols=['Start Time','Start Station ID','Stop Time','End Station ID','Trip Duration'], low_memory=False)
OD_2016_12.rename(columns = {"Trip Duration": "duration_sec", "Start Time": "start_date", "Stop Time": "end_date","Start Station ID":"start_station_code","End Station ID":"end_station_code"}, inplace=True)

SD_2016 = OD_2016_123.append(OD_2016_04, ignore_index=True)
SD_2016 = SD_2016.append(OD_2016_05, ignore_index=True)
SD_2016 = SD_2016.append(OD_2016_06, ignore_index=True)
SD_2016 = SD_2016.append(OD_2016_07, ignore_index=True)
SD_2016 = SD_2016.append(OD_2016_08, ignore_index=True)
SD_2016 = SD_2016.append(OD_2016_09, ignore_index=True)
SD_2016 = SD_2016.append(OD_2016_10, ignore_index=True)
SD_2016 = SD_2016.append(OD_2016_11, ignore_index=True)
SD_2016 = SD_2016.append(OD_2016_12, ignore_index=True)

SD_2016 = SD_2016.sort_values('start_date')
SD_2016.reset_index(drop=True, inplace=True)
SD_2016.head()
SD_2016.tail()

Stations.head()

### 2.1.1. Let's sort it by the total useage duration

SD_2016.end_station_code = SD_2016.end_station_code.astype(int)
StartUsedSorted = pd.DataFrame(SD_2016.groupby(by=['start_station_code'])['duration_sec'].sum())
StartUsedSorted.columns = ['Total duration seconds']
StartUsedSorted = StartUsedSorted.sort_values(by = 'Total duration seconds', ascending=False)
StartUsedSorted.head()

### 2.1.2. Let's sort it by the total used times

StartSorted = SD_2016.groupby(by=['start_station_code'])['start_date'].agg({'Count': np.size})
StartSorted['Count'] = StartSorted.Count.astype(int)
StartSorted = StartSorted.sort_values(by = 'Count', ascending=False)
StartSorted.head()

### 2.1.3. Let's plot useage duration distribution and Toal useage times

f, axes = plt.subplots(2, 1, figsize=(20,10))
plt.sca(axes[0])
TopStartUsedStation = np.array(StartUsedSorted.head(20).index)
TopStartUsedStationData = SD_2016[SD_2016['start_station_code'].isin(TopStartUsedStation)]
sns.lvplot(data=TopStartUsedStationData,x='start_station_code', y='duration_sec',order=TopStartUsedStation,palette="ocean")
plt.title('The longest useage duration of start station in 2016', fontsize = 18)

plt.sca(axes[1])
TopStartUsed = np.array(StartSorted.head(20).index)
TopStartUsedData = SD_2016[SD_2016['start_station_code'].isin(TopStartUsed)]
sns.countplot(data= TopStartUsedData, x ='start_station_code',order=TopStartUsed,palette="ocean")
plt.title('The most useage times of start station in 2016', fontsize = 18)
plt.show()

### 2.1.4. Let's see what is the most useless station based on total used duration

StartUsedSorted_r = StartUsedSorted.sort_values(by = 'Total duration seconds', ascending=True)
StartUsedSorted_r.head()

### 2.1.5. Let's see what is the most useless station based on total used times

StartSorted_r = StartSorted.sort_values(by = 'Count', ascending=True)
StartSorted_r.head()

### 2.1.6. Let's plot the imformation of most useless station

f, axes = plt.subplots(2, 1, figsize=(20,10))
plt.sca(axes[0])
LowStartUsedStation = np.array(StartUsedSorted_r.head(20).index)
LowStartUsedStationData = SD_2016[SD_2016['start_station_code'].isin(LowStartUsedStation)]
vis2 = sns.lvplot(data=LowStartUsedStationData,x='start_station_code', y='duration_sec', order=LowStartUsedStation, palette="ocean_r")
plt.title('The shortest useage duration of start station in Jan.2017', fontsize = 18)

plt.sca(axes[1])
LowStartUsed = np.array(StartSorted_r.head(20).index)
LowStartUsedData = SD_2016[SD_2016['start_station_code'].isin(LowStartUsed)]
sns.countplot(data= LowStartUsedData, x ='start_station_code',order=LowStartUsed,palette="ocean_r")
plt.title('The Least usage times of start station in Jan.2017', fontsize = 18)
plt.show()

### 2.2. Analysing Based on End Station
### 2.2.1. Let's sort it by the total useage duration
EndUsedSorted = pd.DataFrame(SD_2016.groupby(by=['end_station_code'])['duration_sec'].sum())
EndUsedSorted.columns = ['Total duration seconds']
EndUsedSorted = EndUsedSorted.sort_values(by = 'Total duration seconds', ascending=False)
EndUsedSorted.head()

### 2.2.2. Let's sort it by the total used times
EndSorted = SD_2016.groupby(by=['end_station_code'])['start_date'].agg({'Count': np.size})
EndSorted['Count'] = EndSorted.Count.astype(int)
EndSorted = EndSorted.sort_values(by = 'Count', ascending=False)
EndSorted.head()

### 2.2.3. Let's plot useage duration distribution and Toal useage times
f, axes = plt.subplots(2, 1, figsize=(20,10))
plt.sca(axes[0])
TopEndUsedStation = np.array(EndUsedSorted.head(20).index)
TopEndUsedStationData = SD_2016[SD_2016['end_station_code'].isin(TopEndUsedStation)]
sns.lvplot(data=TopEndUsedStationData,x='end_station_code', y='duration_sec',order=TopEndUsedStation, palette="summer")
plt.title('The longest useage duration of end station in Jan.2017', fontsize = 18)

plt.sca(axes[1])
TopEndUsed = np.array(EndSorted.head(20).index)
TopEndUsedData = SD_2016[SD_2016['end_station_code'].isin(TopEndUsed)]
sns.countplot(data= TopEndUsedData, x ='end_station_code',order=TopEndUsed, palette="summer")
plt.title('The most useage times of end station in Jan.2017', fontsize = 18)
plt.show()

### 2.2.4. Let's see what is the most useless station based on total used duration
EndUsedSorted_r = EndUsedSorted.sort_values(by = 'Total duration seconds', ascending=True)
EndUsedSorted_r.head()

### 2.2.5. Let's see what is the most useless station based on total used times
EndSorted_r = EndSorted.sort_values(by = 'Count', ascending=True)
EndSorted_r.head()

### 2.2.6. Let's Plot the Imformation of Most Useless Station
f, axes = plt.subplots(2, 1, figsize=(20,10))
plt.sca(axes[0])
LowEndUsedStation = np.array(EndUsedSorted_r.head(20).index)
LowEndUsedStationData = SD_2016[SD_2016['end_station_code'].isin(LowEndUsedStation)]
sns.lvplot(data=LowEndUsedStationData,x='end_station_code', y='duration_sec',order=LowEndUsedStation, palette="summer_r")
plt.title('The Shortest useage duration of end station in 2016', fontsize = 18)

plt.sca(axes[1])
LowEndUsed = np.array(EndSorted_r.head(20).index)
LowEndUsedData = SD_2016[SD_2016['end_station_code'].isin(LowEndUsed)]
sns.countplot(data= LowEndUsedData, x ='end_station_code',order=LowEndUsed, palette="summer_r")
plt.title('The least useage times of end station in 2016', fontsize = 18)
plt.show()

###2.3 Let's Plot the Imformation of Most Useful Station based on map
SD_2016_05 = OD_2016_05
SD_2016_05.start_station_code = SD_2016_05.start_station_code.astype(int)
StartUsedSorted_05 = pd.DataFrame(SD_2016_05.groupby(by=['start_station_code'])['duration_sec'].sum())
StartUsedSorted_05.columns = ['Total duration seconds']
StartUsedSorted_05 = StartUsedSorted_05.sort_values(by = 'Total duration seconds', ascending=False)
StartUsedSorted_05.head()

StartUsedSorted_05['start_station_code'] = StartUsedSorted_05.index
Full2017 = StartUsedSorted_05.merge(Stations, left_on = 'start_station_code', right_on='code' )
station_useage = Full2017['Total duration seconds'].values

plt.figure(figsize=(11,11))

colors_choice = ['red', 'yellow', 'pink','lightblue', 'green']
size_limits = [0, 1000000, 4000000, 7000000, 10000000,15000000]
labels = []
size_choice = [1, 2, 4,7,9]
for i in range(len(size_limits)-1):
    labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 
map = Basemap(projection='lcc', 
            lat_0=40.76,
            lon_0=-73.98,
            resolution='f',
            llcrnrlon=-74.06, llcrnrlat=40.65,
            urcrnrlon=-73.88, urcrnrlat=40.84)
map.drawcoastlines()
map.drawcountries()
map.drawrivers(color='#0099ff')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='w', lake_color='#0099ff')
#map.fillcontinents(rivers_color='aqua')
for index, (code, y,x) in Full2017[['start_station_code', 'latitude', 'longitude']].iterrows():
    x, y = map(x, y)
    isize = [i for i, val in enumerate(size_limits) if val < station_useage[index]]
    ind = isize[-1]
    map.plot(x, y, marker='o', markersize = size_choice[ind], alpha=1, markeredgewidth = 0.8, color = colors_choice[ind], markeredgecolor='k', label = labels[ind])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
key_order = ('0 <.< 1000000', '1000000 <.< 4000000', '4000000 <.< 7000000',
             '7000000 <.< 10000000','10000000 <.< 15000000')
new_label = OrderedDict()
for key in key_order:
    new_label[key] = by_label[key]
plt.legend(new_label.values(), new_label.keys(), loc = 0, prop= {'size':11},
           title='NYC Total Usage Duration of May 2016', frameon = True, framealpha = 1)
plt.show()

## 3 Predict Model
### 3.1 Create data frame
OD_2016_123.start_date = pd.to_datetime(OD_2016_123.start_date, format='%m/%d/%y %H:%M')
OD_2016_04.start_date = pd.to_datetime(OD_2016_04.start_date, format='%m/%d/%Y %H:%M:%S')
OD_2016_05.start_date = pd.to_datetime(OD_2016_05.start_date, format='%m/%d/%y %H:%M')
for i in range(4):
    j = i+6
    j = '0{}'.format(j)
    s = str(j)
    k = 'OD_2016_'+ s
    #print (k)
    k = eval(k)
    k.start_date = pd.to_datetime(k.start_date, format='%m/%d/%Y %H:%M:%S')
OD_2016_10.start_date = pd.to_datetime(OD_2016_10.start_date, format='%Y-%m-%d %H:%M:%S')
OD_2016_11.start_date = pd.to_datetime(OD_2016_11.start_date, format='%Y-%m-%d %H:%M:%S')
OD_2016_12.start_date = pd.to_datetime(OD_2016_12.start_date, format='%Y-%m-%d %H:%M:%S')

OD_2016 = OD_2016_123.append(OD_2016_04, ignore_index=True)
OD_2016 = OD_2016.append(OD_2016_05, ignore_index=True)
OD_2016 = OD_2016.append(OD_2016_06, ignore_index=True)
OD_2016 = OD_2016.append(OD_2016_07, ignore_index=True)
OD_2016 = OD_2016.append(OD_2016_08, ignore_index=True)
OD_2016 = OD_2016.append(OD_2016_09, ignore_index=True)
OD_2016 = OD_2016.append(OD_2016_10, ignore_index=True)
OD_2016 = OD_2016.append(OD_2016_11, ignore_index=True)
OD_2016 = OD_2016.append(OD_2016_12, ignore_index=True)

OD_2016

#Extract the year, month, and day from start_date
OD_2016['date'] = OD_2016.start_date.dt.date

#Count the total trips of each day
dates = {}
for d in OD_2016.date:
    if d not in dates:
        dates[d] = 1
    else:
        dates[d] += 1
#Create the data frame that will be used for training, with the dictionary we just created.
df = pd.DataFrame.from_dict(dates, orient = "index")
df['date'] = df.index
df['trips'] = df.ix[:,0]
train = df.ix[:,1:3]
train.reset_index(drop = True, inplace = True)

#All sorted now!
train = train.sort_values('date')
train.reset_index(drop=True, inplace=True)

train

### 3.2 Add features to train data 
#Find all of the holidays during our time span
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=train.date.min(), end=train.date.max())

#Find all of the business days in our time span
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
business_days = pd.DatetimeIndex(start=train.date.min(), end=train.date.max(), freq=us_bd)

business_days = pd.to_datetime(business_days, format='%Y/%m/%d').date
holidays = pd.to_datetime(holidays, format='%Y/%m/%d').date

#A 'business_day' or 'holiday' is a date within either of the respected lists.
train['business_day'] = train.date.isin(business_days)
train['holiday'] = train.date.isin(holidays)

train

#Convert True to 1 and False to 0
train.business_day = train.business_day.map(lambda x: 1 if x == True else 0)
train.holiday = train.holiday.map(lambda x: 1 if x == True else 0)

#Convert date to the important features, year, month, weekday (0 = Monday, 1 = Tuesday...)
#We don't need day because what it represents changes every year.
train['year'] = pd.to_datetime(train['date']).dt.year
train['month'] = pd.to_datetime(train['date']).dt.month
train['weekday'] = pd.to_datetime(train['date']).dt.weekday

labels = train.trips
train = train.drop(['trips', 'date'], 1)

train

### 3.3 machine learning proceeding
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error, median_absolute_error

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state = 2)

def scoring(clf):
    scores = cross_val_score(clf, X_train, y_train, cv=15, n_jobs=1, scoring = 'neg_median_absolute_error')
    print (np.median(scores) * -1)

rfr = RandomForestRegressor(n_estimators = 55,
                            min_samples_leaf = 3,
                            random_state = 2)
scoring(rfr)

gbr = GradientBoostingRegressor(learning_rate = 0.12,
                                n_estimators = 150,
                                max_depth = 8,
                                min_samples_leaf = 1,
                                random_state = 2)
scoring(gbr)

dtr = DecisionTreeRegressor(min_samples_leaf = 3,
                            max_depth = 8,
                            random_state = 2)
scoring(dtr)

abr = AdaBoostRegressor(n_estimators = 100,
                        learning_rate = 0.1,
                        loss = 'linear',
                        random_state = 2)
scoring(abr)

import warnings
warnings.filterwarnings("ignore")

random_state = 2
params = {
        'eta': 0.15,
        'max_depth': 6,
        'min_child_weight': 2,
        'subsample': 1,
        'colsample_bytree': 1,
        'verbose_eval': True,
        'seed': random_state,
    }

n_folds = 15 #number of Kfolds
cv_scores = [] #The sum of the mean_absolute_error for each fold.
early_stopping_rounds = 100
iterations = 10000
printN = 50
fpred = [] #stores the sums of predicted values for each fold.

testFinal = xgb.DMatrix(X_test)

kf = KFold(len(X_train), n_folds=n_folds)

for i, (train_index, test_index) in enumerate(kf):
    print('\n Fold %d' % (i+1))
    Xtrain, Xval = X_train.iloc[train_index], X_train.iloc[test_index]
    Ytrain, Yval = y_train.iloc[train_index], y_train.iloc[test_index]
    
    xgtrain = xgb.DMatrix(Xtrain, label = Ytrain)
    xgtest = xgb.DMatrix(Xval, label = Yval)
    watchlist = [(xgtrain, 'train'), (xgtest, 'eval')] 
    
    xgbModel = xgb.train(params, 
                         xgtrain, 
                         iterations, 
                         watchlist,
                         verbose_eval = printN,
                         early_stopping_rounds=early_stopping_rounds
                        )
    
    scores_val = xgbModel.predict(xgtest, ntree_limit=xgbModel.best_ntree_limit)
    cv_score = median_absolute_error(Yval, scores_val)
    print('eval-MSE: %.3f' % cv_score)
    y_pred = xgbModel.predict(testFinal, ntree_limit=xgbModel.best_ntree_limit)
    print(xgbModel.best_ntree_limit)

    if i > 0:
        fpred = pred + y_pred #sum predictions
    else:
        fpred = y_pred
    pred = fpred
    cv_scores.append(cv_score)

xgb_preds = pred / n_folds #find the average values for the predictions
score = np.median(cv_scores)
print('Median error: %.3f' % score)

#Train and make predictions with the best models.
gbr = gbr.fit(X_train, y_train)
dtr = dtr.fit(X_train, y_train)

gbr_preds = gbr.predict(X_test)
dtr_preds = dtr.predict(X_test)

#Weight the top models to find the best prediction
final_preds = dtr_preds*0.4 + gbr_preds*0.6
print ("Daily error of trip count:", median_absolute_error(y_test, final_preds))

y_test.reset_index(drop = True, inplace = True)

### 3.4 Plot the prediction with the actual data
fs = 16
plt.figure(figsize=(16,5))
plt.plot(final_preds,'bo-')
plt.plot(y_test,'rv-')
plt.legend(['Prediction', 'Acutal'])
plt.ylabel("Number of Trips", fontsize = fs)
plt.xlabel("Predicted Date", fontsize = fs)
plt.title("Predicted Values vs Actual Values", fontsize = fs)
plt.show()






