import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from prophet import Prophet  
from pandas.tseries.frequencies import infer_freq



melt = pd.read_csv('//Users/domealbe/Desktop/DATA_SCIENCE/unieuro/luiss-businesscase2-market_data.csv', delimiter=';')
melt = melt.sort_values(by=["PRODUCT_GROUP", 'BRAND', "YEARMONTH"], ascending=[True, True, True])
melt= melt.sort_values(by=["YEARMONTH"], ascending=[True])
melt['YEARMONTH'] = pd.to_datetime(melt['YEARMONTH'], format='%Y%m')
# melt['YEARMONTH'] = melt['YEARMONTH'].dt.strftime('%Y-%m')

df_all = melt[['YEARMONTH','SALES_TOTAL','PRODUCT_GROUP']]
df_all = df_all.sort_values(by=['PRODUCT_GROUP','YEARMONTH'])

# creiamo un dictionary contenente tutti i product_group
prodotti = df_all.groupby(['PRODUCT_GROUP'])
dict_product = {name: group for name, group in prodotti}

# scegliamo un product_group, per provarlo su altri settori basta cambiare il nome SMARTPHONES
# con quello di un altro product_group
product = dict_product['SMARTPHONES']
product = product.groupby('YEARMONTH').sum()

# train - test split
train,test = train_test_split(product, train_size=0.85, shuffle=False)

def create_features(df):  
    df['date'] = df.index

'threat covid as holidays off' 
# creiamo la variabile lockdowns che andremo ad utilizzare nel modello prophet 
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03', 'lower_window': 0, 'ds_upper': '2020-05'},
])

for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days


# creiamo una colonna con le date che andremo a rinominare tra poco
create_features(train)
create_features(test)

# rinominiamo le colonne in ds e y perche prophet per funzionare vuole le colonne in questo modo
train = train.rename(columns={'date':'ds',
   'SALES_TOTAL':'y'})

test= test.rename(columns={'date':'ds',
   'SALES_TOTAL':'y'})

'model'
# define the grid search
from sklearn.model_selection import ParameterGrid
# this grid search will take a lot of time
param_grid = {
    'changepoint_prior_scale':[0.05,0.1,0.5,1,3,5,8,10,12],
    'n_changepoints':[8,10,12,20,25,30,40],
    'seasonality_prior_scale':[0.05,0.1,0.4,1,2,],
    'holidays_prior_scale': [0.1,0.5, 1, 5,10],
    'yearly_seasonality':[8,10,12,15,20,21,25,27,30,35,37,40,'auto'],
    'seasonality_mode': ['additive', 'multiplicative'],
    'growth': ['linear','flat']
}
grid = ParameterGrid(param_grid)
# Creo un DataFrame vuoto per salvare i risultati delle diverse configurazioni di modelli
model_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])

# Loop attraverso tutte le possibili configurazioni di iperparametri
for p in grid:   
    # start the grid search to find the best parameters
    train_model =Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                         n_changepoints = p['n_changepoints'],
                         seasonality_prior_scale = p['seasonality_prior_scale'],
                         holidays_prior_scale = p['holidays_prior_scale'],
                         seasonality_mode = p['seasonality_mode'],
                         growth = p['growth'],
                         yearly_seasonality = p['yearly_seasonality'], 
                         interval_width=0.95)
    
    # fit the model on the train set
    train_model.fit(train)
    
    # predict on th test set
    forecast = train_model.predict(test)
    
    # Calcola il MAPE tra le previsioni e le osservazioni reali
    MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
   
    # Aggiunge i risultati correnti al DataFrame dei risultati dei modelli
    model_parameters = model_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
    
    # Ordina i modelli per MAPE e reimposta l'indice
    parameters = model_parameters.sort_values(by=['MAPE'])
    parameters = parameters.reset_index(drop=True)
    
    # Stampo la configurazione di iperparametri con il miglior MAPE
    print(parameters['Parameters'][0])

#define the model
model = Prophet('mettere i migliori parametri trovati utilizzando la grid search')

#fit the model
model = model.fit(train)
# predict
prediction = model.predict(test)

# benchmark
test['mean'] = train['y'].mean()  

# calculate the mape
from sklearn.metrics import mean_absolute_percentage_error
MAPE = mean_absolute_percentage_error(test['y'], prediction['yhat'])
mape_media = mean_absolute_percentage_error(test['y'], test['mean'])

'grafico predizione sul test'

plt.xlim(['2015-01', '2022-03'])

import matplotlib.dates as mdates

f, ax = plt.subplots(figsize=(25, 10), dpi=500)
plt.title('TOTAL SALES MODEL PREDICTION PER PRODUCT GROUP : "PRODUCT_GROUP" ', fontsize=15)

prophet_data = pd.concat([train, test], axis=0)

plt.plot(prophet_data.index, prophet_data['y'], label='TOTAL SALES')
plt.plot(test.index, prediction['yhat'], label='TOTAL SALES PREDICTED BY THE MODEL')
plt.plot(test.index, test['mean'], label='TOTAL SALES MEAN BENCHMARK')

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
plt.xlim(['2015-01', '2022-03'])

plt.show()

'FUTURE'
'create future'
# create a list of months from February 2022 to February 2023
months = pd.date_range(start='2022-03-01', end='2023-02-01', freq='MS')
# create a DataFrame with the months as the index and an empty column
future = pd.DataFrame(index=months, columns=['SALES_OFFLINE'])

#questa funzione serve a creare una colonna con la data 
def create_features(df):
    df['date'] = df.index
    
# creiamo una colonna con le date che andremo a rinominare tra poco
create_features(product)
create_features(future)

product = product.rename(columns={'date':'ds',
   'SALES_TOTAL':'y'})

future= future.rename(columns={'date':'ds',
   'SALES_TOTAL':'y'})

# fit the model
model = model.fit(product)

# predict thefuture
prediction = model.predict(future)

plt.xlim(['2015-01', '2023-03'])
import matplotlib.dates as mdates
f, ax = plt.subplots(figsize=(25, 10), dpi=500)
plt.title('TOTAL SALES MODEL PREDICTION PER PRODUCT GROUP: "PRODUCT_GROUP"', fontsize=15)

# merge the data
prophet_data = pd.concat([product, future], axis=0)
plt.plot(product.index, product['y'], label='TOTAL SALES')
plt.plot(future.index,prediction['yhat'], label = 'TOTAL SALES PREDICTED BY THE MODEL')

plt.legend(fontsize=15)

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
plt.xlim(['2015-01', '2023-03'])

plt.show()
    
# store the sum of the predicted sales from 03-22 to 02-23
predicted_sales = prediction['yhat'].sum()

# store the sum of the sales from 03-21 to 02-22
previous_year_sales = product[product.index >= '2021-03-01']
previous_year_sales = previous_year_sales['SALES_TOTAL'].sum()

# calculate the predicted sales percentage change 
percentage_change = ((predicted_sales - previous_year_sales) / previous_year_sales) * 100














