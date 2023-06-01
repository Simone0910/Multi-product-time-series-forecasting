from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

melt = pd.read_csv('//Users/domealbe/Desktop/DATA_SCIENCE/unieuro/luiss-businesscase2-market_data.csv', delimiter=';')
melt = melt.sort_values(by=["PRODUCT_GROUP", 'BRAND', "YEARMONTH"], ascending=[True, True, True])
melt= melt.sort_values(by=["YEARMONTH"], ascending=[True])
melt['YEARMONTH'] = pd.to_datetime(melt['YEARMONTH'], format='%Y%m')
# melt['YEARMONTH'] = melt['YEARMONTH'].dt.strftime('%Y-%m')

df = melt[['YEARMONTH','SALES_OFFLINE','SALES_ONLINE','BRAND','PRODUCT_GROUP','SECTOR']]
df = df.sort_values(by=['PRODUCT_GROUP','BRAND','YEARMONTH'])

##################
################## apple smartphone
gruppi = df.groupby((df['BRAND'] != df['BRAND'].shift()).cumsum())
list_df_slices = [gruppo for _, gruppo in gruppi]


gruppi = df.groupby(['BRAND', 'PRODUCT_GROUP'])
dict_df_slices = {name: group for name, group in gruppi}


apple_smartphone = dict_df_slices['APPLE', 'SMARTPHONES']
apple_smartphone.set_index('YEARMONTH', inplace=True)

# filtering only  the column we need
apple_smartphone_OFF = apple_smartphone.filter(['SALES_OFFLINE', 'PRODUCT_GROUP', 'BRAND', 'SECTOR'])
apple_smartphone_OFF = apple_smartphone_OFF.drop(['PRODUCT_GROUP', 'BRAND', 'SECTOR'], axis = 1)

# splitto
apple_smartphone_OFF_train = apple_smartphone_OFF[apple_smartphone_OFF.index < '2021-02-01']
apple_smartphone_OFF_test = apple_smartphone_OFF[apple_smartphone_OFF.index >= '2021-02-01']


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]
         


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, 1], predictions

# load the dataset
series = apple_smartphone_OFF
values = series.values
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=17)
# evaluate
mae, y, yhat = walk_forward_validation(data, 13)

# salviamo le predizioni nel df
apple_smartphone_OFF_test['prediction'] = yhat


# plottiamo
plt.plot(apple_smartphone_OFF_test.index, apple_smartphone_OFF_test['SALES_OFFLINE'])
plt.plot(apple_smartphone_OFF_test.index, apple_smartphone_OFF_test['prediction'])
plt.legend()
plt.show()


from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(apple_smartphone_OFF_test['SALES_OFFLINE'], apple_smartphone_OFF_test['prediction'])
print('MAPE: ', mape)


















