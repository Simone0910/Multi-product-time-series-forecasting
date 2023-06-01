import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prophet
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

melt = pd.read_csv('//Users/domealbe/Desktop/DATA_SCIENCE/unieuro/luiss-businesscase2-market_data.csv', delimiter=';')
melt = melt.sort_values(by=["PRODUCT_GROUP", 'BRAND', "YEARMONTH"], ascending=[True, True, True])
melt= melt.sort_values(by=["YEARMONTH"], ascending=[True])
melt['YEARMONTH'] = pd.to_datetime(melt['YEARMONTH'], format='%Y%m')
# melt['YEARMONTH'] = melt['YEARMONTH'].dt.strftime('%Y-%m')

df = melt[['YEARMONTH','SALES_OFFLINE','SALES_ONLINE','BRAND','PRODUCT_GROUP','SECTOR']]
df = df.sort_values(by=['PRODUCT_GROUP','BRAND','YEARMONTH'])



# splittiamo tutto il dataframe in 3217 time series differenti in base ad ogni product group di ogni brand
gruppi = df.groupby(['BRAND', 'PRODUCT_GROUP'])
dict_df_slices = {name: group for name, group in gruppi}

# prendiamo la serie temporale APPLE SMARTPHONES
apple_smartphone = dict_df_slices['APPLE', 'SMARTPHONES']
apple_smartphone.set_index('YEARMONTH', inplace=True)


# filtriamo e droppiamo le colonne che non ci interessano
apple_smartphone_OFF = apple_smartphone.filter(['SALES_OFFLINE', 'PRODUCT_GROUP', 'BRAND', 'SECTOR'])
apple_smartphone_OFF = apple_smartphone_OFF.drop(['PRODUCT_GROUP', 'BRAND', 'SECTOR'], axis = 1)


# splitto
apple_smartphone_OFF_train = apple_smartphone_OFF[apple_smartphone_OFF.index < '2021-02-01']
apple_smartphone_OFF_test = apple_smartphone_OFF[apple_smartphone_OFF.index >= '2021-02-01']


###################################################################################################
###################################################################################################
###################################################################################################
'MODELLO ARIMA INIZIALE, DATI NON STAZIONARI'
                      

#funzione per vedere se la serie è stationary
from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
     dftest = adfuller(apple_smartphone_OFF, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
ad_test(apple_smartphone_OFF['SALES_OFFLINE'])



from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
#stepwise 
stepwise_fit = auto_arima(apple_smartphone_OFF_train['SALES_OFFLINE'], trace=True, suppress_warnings=True)

# definiamo il modello con l'order datoci dalla stepwise
model = ARIMA(apple_smartphone_OFF_train['SALES_OFFLINE'], order=(2,1,3))

# fittiamo il modello sul train
model_fit = model.fit()

# predicta i valori
apple_smartphone_OFF_test['SALES_OFFLINE_forecast'] = model_fit.predict(start=apple_smartphone_OFF_test.index[0], end=apple_smartphone_OFF_test.index[-1], dynamic=True)

#calculate the mape
mape = mean_absolute_percentage_error(apple_smartphone_OFF_test['SALES_OFFLINE'], apple_smartphone_OFF_test['SALES_OFFLINE_forecast'])   
print('MAPE:', mape)

# plot the prediction on the test set
plt.plot(apple_smartphone_OFF_test['SALES_OFFLINE'],label = 'SALES_OFFLINE')
plt.plot(apple_smartphone_OFF_test['SALES_OFFLINE_forecast'],label = 'SALES_FORECAST')
plt.xticks(rotation=90)
plt.legend()




###################################################################################################
###################################################################################################
###################################################################################################
'MODELLO ARIMA CON DATI STAZIONARI --> TRASFORMAZIONE LOG E SHIFT'


#log sul trian
apple_smartphone_OFF_logScale_train = np.log(apple_smartphone_OFF_train)

#log sul test
apple_smartphone_OFF_logScale_test = np.log(apple_smartphone_OFF_test)

# utilizziamo la differencing per rendere statonary il train set e il test set
datasetLogDiffShifting_train = apple_smartphone_OFF_logScale_train.diff()
datasetLogDiffShifting_train.dropna(inplace=True)

#differencing sul test set
datasetLogDiffShifting_test = apple_smartphone_OFF_logScale_test.diff()
datasetLogDiffShifting_test.dropna(inplace=True)

# define the model
model = ARIMA(datasetLogDiffShifting_train, order=(2,1,3))

# Addestra il modello sui dati di training
model_fit = model.fit()

# predicta i valori
prediction = model_fit.predict(start=datasetLogDiffShifting_test.index[0], end=datasetLogDiffShifting_test.index[-1], dynamic=True)

# mape sulla serie log shift
mape_log_shift = mean_absolute_percentage_error(datasetLogDiffShifting_test['SALES_OFFLINE'], prediction)


'plot dei dati ancora non riportati alla normalità'
plt.plot(datasetLogDiffShifting_test.index,datasetLogDiffShifting_test['SALES_OFFLINE'], label='Valori osservati')
plt.plot(datasetLogDiffShifting_test.index, prediction, label='Valori predetti')
plt.legend()
plt.show()

'facciamo tornare i valori sulla scala originale'
# prendiamo il primo valore della serie che andremo ad aggiungere a tutti i valori cosi torniamo
# alla normalità dopo il differencing
value_to_add= apple_smartphone_OFF_logScale_test['SALES_OFFLINE'].iloc[0]

# li facciamo tornare alla normalità prima facendo il cum sum
datasetLogDiffShifting_test_cum= datasetLogDiffShifting_test['SALES_OFFLINE'].cumsum()  + value_to_add

prediction_cum= prediction.cumsum() +value_to_add


#facciamo l'esponente per riportare i valori completamente alla normalità
datasetLogDiffShifting_test_exp =  np.exp(datasetLogDiffShifting_test_cum)
prediction_exp = np.exp(prediction_cum)

# calcoliamo il mape con i valori riportati alla normalità
mape_normalized = mean_absolute_percentage_error(datasetLogDiffShifting_test_exp, prediction_exp)
print('MAPE:', mape_normalized)

'plot dei dati riportati alla normalità'
plt.plot(datasetLogDiffShifting_test_exp.index,datasetLogDiffShifting_test_exp, label='Valori osservati')
plt.plot(datasetLogDiffShifting_test_exp.index,prediction_exp, label='Valori predetti')
plt.legend()
plt.show()



###################################################################################################
###################################################################################################
###################################################################################################
'MODELLO ARIMA CON STAGIONALITA e GESTIONE DEL COVID'

#seasonal stepwise : modello suggerito 2, 1, 3
import statsmodels.api as sm
# Load data
data = apple_smartphone_OFF_train['SALES_OFFLINE']

# Set frequency of time series to monthly
data = data.asfreq('MS')

# Define search ranges for p, d, q and seasonal P, D, Q
p = range(0, 4)
d = range(0, 2)
q = range(0, 4)

# Create list of all possible combinations of p, d, q values
pdq = [(p_, d_, q_) for p_ in p for d_ in d for q_ in q]

# Define evaluation metric
ic = 'aic'

# Run evaluation
results = []
for params in pdq:
    try:
        model = sm.tsa.ARIMA(data, order=params)
        output = model.fit()
        results.append({'params': params, 'aic': output.aic, 'bic': output.bic})
    except:
        continue
        
# Convert results to pandas DataFrame
results_df = pd.DataFrame(results)

# Find best parameter combination based on evaluation metric
best_params = results_df.loc[results_df[ic].idxmin(), 'params']

# Print best parameters and evaluation metric value
print('Best parameters: ', best_params)
print('Best', ic.upper(), ':', results_df.loc[results_df[ic].idxmin(), ic])


# Creazione di una serie temporale di variabili binarie per escludere le date specifiche come festività
exog = pd.DataFrame({'holiday': np.zeros(len(apple_smartphone_OFF_train))}, 
                    index=apple_smartphone_OFF_train.index)
exog.loc[(exog.index >= '2020-02') & (exog.index <= '2020-05'), 'holiday'] = 1



# define the model using the order of the first stepwise (2,1.3) and the seasonal order found with the 
# seasonal stepwise (5,0,3)
model = ARIMA(apple_smartphone_OFF_train['SALES_OFFLINE'], 
               order=(2,1,3), #top 213 #poi 113, 101, 
              seasonal_order=(5,0,3,12), #top 503
              exog=exog
              )
# fit the model
model_fit = model.fit()

# predict
apple_smartphone_OFF_test['SALES_OFFLINE_forecast'] = model_fit.predict(start=apple_smartphone_OFF_test.index[0], 
                                                                       end=apple_smartphone_OFF_test.index[-1], 
                                                                       exog=np.zeros(len(apple_smartphone_OFF_test)), 
                                                                       dynamic=True, 
                                                                       typ='levels')

# prediction on test plot 
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
# plt.plot(apple_smartphone_OFF_train['SALES_OFFLINE'], label='Train')
plt.plot(apple_smartphone_OFF_test['SALES_OFFLINE'], label='SAlES_OFFLINE')
plt.plot(apple_smartphone_OFF_test['SALES_OFFLINE_forecast'], label='SAlES_OFFLINE_FORECAST')
plt.title('ARIMA Seasonal Forecast')
plt.xlabel('Date')
plt.ylabel('Sales Offline')
plt.xticks(rotation=45)
plt.legend()
plt.show()


from sklearn.metrics import mean_absolute_percentage_error
mape_seasonal_covid = mean_absolute_percentage_error(apple_smartphone_OFF_test['SALES_OFFLINE'], apple_smartphone_OFF_test['SALES_OFFLINE_forecast'])

'predict the future'

# create a list of months from February 2022 to February 2023
months = pd.date_range(start='2022-03-01', end='2023-02-01', freq='MS')
# create a DataFrame with the months as the index and an empty column
future = pd.DataFrame(index=months, columns=['SALES_OFFLINE'])


# definiamo il modello
arima_future = ARIMA(apple_smartphone_OFF['SALES_OFFLINE'], 
               order=(2,1,3), 
              seasonal_order=(5,0,3,12), 
              exog=exog
              )

# fittiamo il modello
arima_future = arima_future.fit()

  
# Make predictions
future['SALES_ONLINE_forecast'] = arima_future.predict(start=future.index[0],
                                     end=future.index[-1],  
                                     dynamic=True, 
                                     typ='levels')



#  plot the future prediction
plt.figure(figsize=(10,10))
plt.plot(apple_smartphone_OFF.index, apple_smartphone_OFF['SALES_OFFLINE'],label = 'SALES_OFFLINE')
plt.plot(future.index , future['SALES_OFFLINE_forecast'], label='Forecast')
plt.title('ARIMA Seasonal Forecast')
plt.xlabel('Date')
plt.ylabel('Sales Offline')
plt.xticks(rotation=45)
plt.legend()
plt.show()
















