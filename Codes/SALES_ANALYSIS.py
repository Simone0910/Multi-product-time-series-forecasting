#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:54:05 2023

@author: domealbe
"""


import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

melt = pd.read_csv('//Users/domealbe/Desktop/DATA_SCIENCE/unieuro/luiss-businesscase2-market_data.csv', delimiter=';')
melt = melt.sort_values(by=["PRODUCT_GROUP", 'BRAND', "YEARMONTH"], ascending=[True, True, True])
melt= melt.sort_values(by=["YEARMONTH"], ascending=[True])
melt['YEARMONTH'] = pd.to_datetime(melt['YEARMONTH'], format='%Y%m')
# melt['YEARMONTH'] = melt['YEARMONTH'].dt.strftime('%Y-%m')

df = melt[['YEARMONTH','SALES_OFFLINE','SALES_ONLINE','BRAND','PRODUCT_GROUP','SECTOR']]
df = df.sort_values(by=['PRODUCT_GROUP','BRAND','YEARMONTH'])

apple_smartphone_OFF = df.groupby(['YEARMONTH'])['SALES_OFFLINE'].sum().reset_index()
apple_smartphone_OFF.set_index('YEARMONTH', inplace=True)


# splitto
apple_smartphone_OFF_train = apple_smartphone_OFF[apple_smartphone_OFF.index < '2021-02-01']
apple_smartphone_OFF_test = apple_smartphone_OFF[apple_smartphone_OFF.index >= '2021-02-01']






####################################################################################
####################################################################################
####################################################################################
####################################################################################
                      # MODELLO PROPHET



sales_per_month= df.groupby('YEARMONTH').sum()
sales_per_month = sales_per_month.filter(['SALES_OFFLINE'], axis=1)



from prophet import Prophet
train = sales_per_month[sales_per_month.index < '2021-02-01']
test = sales_per_month[sales_per_month.index >= '2021-02-01']



def create_features(df):
    df['date'] = df.index
    
create_features(train)
create_features(test)


prophet_train= train.rename(columns={'date':'ds',
                     'SALES_OFFLINE':'y'})

prophet_test = test.rename(columns={'date':'ds',
                     'SALES_OFFLINE':'y'})



'threat covid as holidays off' 

lockdowns = pd.DataFrame([{'holiday': 'lockdown_1', 'ds': '2020-03', 'lower_window': 0, 'ds_upper': '2020-05'},])



for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days

#best params are 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'multiplicative'

'model'
model = Prophet(yearly_seasonality=37,holidays=lockdowns,changepoint_prior_scale=10
                ,seasonality_prior_scale= 0.1, growth = 'flat')
# fit the model
model = model.fit(prophet_train)
prediction = model.predict(prophet_test)

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(prophet_test['y'], prediction['yhat'], squared=False)

RMSE = np.sqrt(MSE)

from sklearn.metrics import mean_squared_log_error
MSLE = mean_squared_log_error(prophet_test['y'], prediction['yhat'], squared=False)

from sklearn.metrics import mean_absolute_percentage_error
MAPE = mean_absolute_percentage_error(prophet_test['y'], prediction['yhat'])

from sklearn.metrics import r2_score
r2 = r2_score(prophet_test['y'], prediction['yhat'])

print('mape:', MAPE)


f, ax = plt.subplots(figsize=(20, 10))
plt.title('Prophet treating covid as holidays off')
plt.plot(prophet_train.index, prophet_train['y'], label = 'actuals')
plt.plot(prophet_test.index, prophet_test['y'], label = 'actuals')
plt.plot(prophet_test.index,prediction['yhat'], label = 'predicted')
plt.legend()
print('mape:', MAPE)



import matplotlib.dates as mdates

f, ax = plt.subplots(figsize=(20, 10))
plt.title('TOTAL SALES_OFFLINE MODEL')

prophet_data = pd.concat([prophet_train, prophet_test], axis=0)

plt.plot(prophet_data.index, prophet_data['y'], label='SALES_OFFLINE')
plt.plot(prophet_test.index, prediction['yhat'], label='SALES_OFFLINE_FORECAST')

plt.legend()
print('mape:', MAPE)

# Imposta i numeri interi sull'asse y
plt.ticklabel_format(useOffset=False, style='plain', axis='y')

# Imposta l'intervallo sull'asse y ogni due mesi
months = mdates.MonthLocator(interval=2)
ax.yaxis.set_major_locator(months)

plt.show()




####################################################################################
####################################################################################
####################################################################################
####################################################################################
#                                   BENCHMARK


# # Definizione della funzione personalizzata per calcolare la media e restituire una nuova riga con l'indice "media"
apple_smartphone_OFF_mean = test['SALES_OFFLINE'].copy()

mean = train['SALES_OFFLINE'].mean()

apple_smartphone_OFF_mean = pd.DataFrame(apple_smartphone_OFF_mean)

apple_smartphone_OFF_mean = apple_smartphone_OFF_mean.assign(SALES_OFFLINE_MEAN=mean)

apple_smartphone_OFF_mean = apple_smartphone_OFF_mean.drop('SALES_OFFLINE', axis=1)


from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(test['SALES_OFFLINE'], apple_smartphone_OFF_mean)

print('MAPE MEAN:', mape)



############### PLOTTING BENCHMARK
plt.xlim(['2015-01', '2022-03'])

import matplotlib.dates as mdates

f, ax = plt.subplots(figsize=(25, 10) )
plt.title('TOTAL SALES_OFFLINE MEAN PREDICTION', fontsize=15)

prophet_data = pd.concat([train, test], axis=0)

plt.plot(prophet_data.index, prophet_data['SALES_OFFLINE'], label='SALES_OFFLINE')
plt.plot(prophet_test.index, apple_smartphone_OFF_mean['SALES_OFFLINE_MEAN'], label='SALES_OFFLINE_MEAN_FORECAST')

plt.legend()
print('MAPE MODEL:', MAPE)

# Imposta la frequenza degli intervalli sull'asse x
locator = mdates.MonthLocator(interval=2)
ax.xaxis.set_major_locator(locator)

# Personalizza il formato della data
formatter = mdates.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(formatter)

# Ruota le etichette delle date di 90 gradi
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

# Imposta i numeri interi sull'asse y
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.grid(True, alpha=0.25)  # Aggiunta della griglia

# Imposta i nomi degli assi x e y
plt.xlabel('YEAR MONTH', fontsize=15)
plt.ylabel('SALES', fontsize=15)

# Imposta i limiti dell'asse x
plt.xlim(['2014-12', '2022-03'])

plt.show()







####################################################################################
####################################################################################
####################################################################################
####################################################################################
#                                PLOTTING MODEL

plt.xlim(['2015-01', '2022-03'])

import matplotlib.dates as mdates

f, ax = plt.subplots(figsize=(25, 10))
plt.title('TOTAL SALES_OFFLINE MODEL PREDICTION', fontsize=15)

prophet_data = pd.concat([prophet_train, prophet_test], axis=0)

plt.plot(prophet_data.index, prophet_data['y'], label='SALES OFFLINE')
plt.plot(prophet_test.index, prediction['yhat'], label='SALES OFFLINE PREDICTED BY THE MODEL')
plt.plot(prophet_test.index, apple_smartphone_OFF_mean['SALES_OFFLINE_MEAN'], label='SALES OFFLINE MEAN BENCHMARK')

plt.legend(fontsize=15)
print('mape:', MAPE)

# Imposta la frequenza degli intervalli sull'asse x
locator = mdates.MonthLocator(interval=2)
ax.xaxis.set_major_locator(locator)

# Personalizza il formato della data
formatter = mdates.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(formatter)

# Ruota le etichette delle date di 90 gradi
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

# Imposta i numeri interi sull'asse y
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.grid(True, alpha=0.25)  # Aggiunta della griglia

# Imposta i nomi degli assi x e y
plt.xlabel('YEAR MONTH', fontsize=15)
plt.ylabel('SALES', fontsize=15)

# Imposta i limiti dell'asse x
plt.xlim(['2014-12', '2022-03'])

plt.show()


####################################################################################
####################################################################################
####################################################################################
####################################################################################


def create_features(df):
    df['date'] = df.index
    'create future'
    
# create a list of months from February 2022 to February 2023
months = pd.date_range(start='2022-03-01', end='2023-02-01', freq='MS')
# create a DataFrame with the months as the index and an empty column
future = pd.DataFrame(index=months, columns=['SALES_OFFLINE'])


create_features(sales_per_month)
create_features(future)

# rinominiamo le colonne in ds e y perche prophet per funzionare vuole le colonne in questo modo
sales_per_month = sales_per_month.rename(columns={'date':'ds',
               'SALES_OFFLINE':'y'})

future= future.rename(columns={'date':'ds',
               'SALES_OFFLINE':'y'})
  
model = Prophet(yearly_seasonality=37,holidays=lockdowns,changepoint_prior_scale=10
          ,seasonality_prior_scale= 0.1, growth = 'flat')
  # fit the model
model = model.fit(sales_per_month)
  
prediction = model.predict(future)

    
    
# plt.xlim(['2015-01', '2022-03'])


####################################################################################
####################################################################################
####################################################################################
####################################################################################
#                                PLOTTING FORECAST
import matplotlib.dates as mdates

f, ax = plt.subplots(figsize=(25, 10) )# PER ALTA DEIFNIZIONE USA ,dpi=300
plt.title('TOTAL SALES_OFFLINE MODEL PREDICTION', fontsize=15)

# prophet_data = pd.concat([prophet_train, prophet_test], axis=0)

plt.plot(sales_per_month.index, sales_per_month['y'], label='SALES OFFLINE')
plt.plot(future.index, prediction['yhat'], label='SALES OFFLINE PFORECAST')

plt.legend(fontsize=15)
print('mape:', MAPE)

# Imposta la frequenza degli intervalli sull'asse x
locator = mdates.MonthLocator(interval=2)
ax.xaxis.set_major_locator(locator)

# Personalizza il formato della data
formatter = mdates.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(formatter)

# Ruota le etichette delle date di 90 gradi
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

# Imposta i numeri interi sull'asse y
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.grid(True, alpha=0.25)  # Aggiunta della griglia

# Imposta i nomi degli assi x e y
plt.xlabel('YEAR MONTH', fontsize=15)
plt.ylabel('SALES', fontsize=15)

# Imposta i limiti dell'asse x
plt.xlim(['2014-12', '2023-03'])

plt.show()


####################################################################################
####################################################################################
####################################################################################
####################################################################################
#               VARIAZIONE PERCENTUALE ANNI

prediction['yhat']
sales_per_month['y']

sales_per_year = sales_per_month.groupby(sales_per_month.index.year)['y'].pct_change().shift(-12)


sales_per_year = sales_per_month.resample('Y').sum()  # somma le vendite per ogni anno
sales_per_year['pct_change'] = sales_per_year['y'].pct_change() * 100  # calcola la variazione percentuale annuale

# sales_per_year.


# sales_per_month['anno_marzo'] = (sales_per_month.index + pd.DateOffset(months=2)).year



sales_per_month= df.groupby('YEARMONTH').sum()
sales_per_month = sales_per_month.filter(['SALES_OFFLINE'], axis=1)


sales_per_month = sales_per_month.drop(sales_per_month.index[0])
sales_per_month = sales_per_month.drop(sales_per_month.index[0])




# Creazione di un DataFrame di esempio con 84 mesi nell'index
idx = pd.date_range(start='2015-03-01', periods=84)

# Raggruppamento dei dati per anno e calcolo delle somme
sales_per_year = sales_per_month.groupby(pd.Grouper(freq='Y')).sum()

# Stampa del nuovo DataFrame
print(sales_per_year)





# Somma dei primi 12 valori
y2015 = sales_per_month.iloc[:12].sum()
y2016 = sales_per_month.iloc[12:24].sum()
y2017 = sales_per_month.iloc[24:36].sum()
y2018 = sales_per_month.iloc[36:48].sum()
y2019 = sales_per_month.iloc[48:60].sum()
y2020 = sales_per_month.iloc[60:72].sum()
y2021 = sales_per_month.iloc[72:84].sum()
y2022 = prediction['yhat'].sum()

# Creazione del DataFrame per ogni anno
df_y2015 = y2015.to_frame('sales')
df_y2016 = y2016.to_frame('sales')
df_y2017 = y2017.to_frame('sales')
df_y2018 = y2018.to_frame('sales')
df_y2019 = y2019.to_frame('sales')
df_y2020 = y2020.to_frame('sales')
df_y2021 = y2021.to_frame('sales')


# Concatenazione dei DataFrame
df_anni = pd.concat([df_y2015, df_y2016, df_y2017, df_y2018, df_y2019, df_y2020, df_y2021])
df_anni.loc['total', 'sales'] = prediction['yhat'].sum()


# Impostazione del nuovo indice
idx = pd.date_range('2015', '2023', freq='Y').strftime('%Y')
df_anni.index = idx


# Calcolo della variazione percentuale rispetto all'anno precedente
df_anni['percent_change'] = df_anni['sales'].pct_change()*100










