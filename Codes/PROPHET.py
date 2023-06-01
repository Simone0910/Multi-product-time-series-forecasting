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


'PROPHET MODEL'
# creiamo la variabile lockdowns che andremo ad utilizzare nel modello prophet 
'threat covid as holidays off' 
lockdowns = pd.DataFrame([ {'holiday': 'lockdown_1', 'ds': '2020-03', 'lower_window': 0, 'ds_upper': '2020-05'},])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days

# creiamo un df vuoto con i mesi da 03-22 a 02-23 che sono i mesi che andremo a predirre
'create future'
# create a list of months from February 2022 to February 2023
months = pd.date_range(start='2022-03-01', end='2023-02-01', freq='MS')
# create a DataFrame with the months as the index and an empty column
future = pd.DataFrame(index=months, columns=['SALES_OFFLINE'])

# definiamo una funzione con cui creiamo una colonna con le date che andremo a rinominare ds
def create_features(df):
    df['date'] = df.index
    
# applichiamo la funzione
create_features(apple_smartphone_OFF_train)
create_features(apple_smartphone_OFF_test)

# definiamo il train
prophet_train= apple_smartphone_OFF_train.rename(columns={'date':'ds',
                     'SALES_OFFLINE':'y'})

# definiamo il test
prophet_test = apple_smartphone_OFF_test.rename(columns={'date':'ds',
                     'SALES_OFFLINE':'y'})

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

# contiamo quanti sono tutti i possibili modelli che proveremo con la grid search
grid = ParameterGrid(param_grid)
cnt = 0
for p in grid:
    cnt = cnt+1

print('Total Possible Models',cnt)

# Creo un DataFrame vuoto per salvare i risultati delle diverse configurazioni di modelli
model_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])

# Loop attraverso tutte le possibili configurazioni di iperparametri
for p in grid:   
    # Inizializzo il modello Prophet con gli iperparametri correnti
    train_model =Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                         n_changepoints = p['n_changepoints'],
                         seasonality_prior_scale = p['seasonality_prior_scale'],
                         holidays_prior_scale = p['holidays_prior_scale'],
                         seasonality_mode = p['seasonality_mode'],
                         growth = p['growth'],
                         yearly_seasonality = p['yearly_seasonality'], 
                         interval_width=0.95)
    
    # Addestra il modello sul set di addestramento
    train_model.fit(prophet_train)
    
    # Esegue le previsioni sul set di test
    forecast = train_model.predict(prophet_test)
    
    # Calcola il MAPE tra le previsioni e le osservazioni reali
    MAPE = mean_absolute_percentage_error(prophet_test['y'],forecast['yhat'])
   
    # Aggiunge i risultati correnti al DataFrame dei risultati dei modelli
    model_parameters = model_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
    
    # Ordina i modelli per MAPE e reimposta l'indice
    parameters = model_parameters.sort_values(by=['MAPE'])
    parameters = parameters.reset_index(drop=True)
    
    # Stampo la configurazione di iperparametri con il miglior MAPE
    print(parameters['Parameters'][0])


# define the model
model = Prophet('mettere i migliori parametri trovati utilizzando la grid search')
# fit the model
model = model.fit(prophet_train)

# predict sul test
prediction = model.predict(prophet_test)

# calcola il mape
from sklearn.metrics import mean_absolute_percentage_error
MAPE = mean_absolute_percentage_error(prophet_test['y'], prediction['yhat'])

# plot the results
fig = plt.figure(figsize=(10, 5))
# plt.plot(apple_smartphone_OFF_train['SALES_OFFLINE'])
plt.plot( prophet_test['y'],label = 'SALES_OFFLINE')
plt.plot(prediction['yhat'],label = 'SALES_FORECAST')
plt.xticks(rotation=90)
plt.legend()

'PREDICT THE FUTURE'
# rinominiamo le colonne nel modo che vuole prophet
df_prophet = df.rename(columns={'date':'ds',
                     'SALES_OFFLINE':'y'})

future= future.rename(columns={'date':'ds',
                     'SALES_OFFLINE':'y'})

# definiamo il modello
model = Prophet('mettere i migliori parametri trovati utilizzando la grid search')

# fit the model
model = model.fit(df_prophet)

# predict future
prophet_prediction = model.predict(future)


# plot the results
fig = plt.figure(figsize=(15, 10))
# plt.plot(apple_smartphone_OFF_train['SALES_OFFLINE'])
plt.plot(df_prophet.index , df_prophet['y'],label = 'SALES_OFFLINE')
plt.plot(future.index,prophet_prediction['yhat'],label = 'SALES_FORECAST')
plt.xticks(rotation=90)
plt.legend()



















