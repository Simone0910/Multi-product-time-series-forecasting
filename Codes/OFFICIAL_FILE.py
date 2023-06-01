import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet  
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from pandas.tseries.frequencies import infer_freq
import statsmodels.api as sm



df = pd.read_csv('//Users/domealbe/Desktop/DATA_SCIENCE/unieuro/luiss-businesscase2-market_data.csv', delimiter=';')
df = df.sort_values(by=["PRODUCT_GROUP", 'BRAND', "YEARMONTH"], ascending=[True, True, True])
df = df.sort_values(by=["YEARMONTH"], ascending=[True])
df['YEARMONTH'] = pd.to_datetime(df['YEARMONTH'], format='%Y%m')
# melt['YEARMONTH'] = melt['YEARMONTH'].dt.strftime('%Y-%m')

df_all = df[['YEARMONTH','SALES_OFFLINE','SALES_ONLINE','BRAND','PRODUCT_GROUP','SECTOR']]
df_all = df_all.sort_values(by=['PRODUCT_GROUP','BRAND','YEARMONTH'])

# splittiamo tutto il dataframe in 3217 time series differenti in base ad ogni product group di ogni brand
gruppi = df_all.groupby(['BRAND', 'PRODUCT_GROUP'])
dict_df_slices = {name: group for name, group in gruppi}


# creiamo il dictionary contenente solo le serie che hanno almeno 12  mesi
dict_model = {}
for name, group in dict_df_slices.items():
    # Check if the dataframe has a valid row coun
    if len(group) >=12:
        dict_model[name] = group
        

#

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

#questa funzione serve a creare una colonna con la data 
def create_features(df):
    df['date'] = df.index

def apply_prophet_arima_model(df,future):
    
      #  settiamo la colonna yearmonth come index
      df.set_index('YEARMONTH', inplace=True)

      # calcola la percentuale di valori 0 nella colonna 'SALES_OFFLINE'
      zero_percentage = sum(df['SALES_OFFLINE'] == 0) / len(df)
      
      # storiamo gli ultimi 12 mesi del df in un variabile
      last_12_sales_offline = df['SALES_OFFLINE'][-12:]
      
      # calcola la percentuale di valori 0 delle vendite negli ultimi 12 mesi
      zero_percentage_12 = sum(last_12_sales_offline == 0) / len(last_12_sales_offline)
      

    # verifica se la percentuale di zeri è minore al 50%, se lo è avvia le predizioni con i modelli
      if zero_percentage < 0.5 or zero_percentage_12 < 0.5:
          
          'ARIMA ON TEST'
          # Load data   
          freq = infer_freq(df.index)
          # interpolate the missing dates

          start_date = df.index[0]
          idx = pd.date_range(start=start_date, end='2022-02-01', freq='MS')
          df = df.reindex(idx, fill_value=None)
          df = df.interpolate()
              
          # split
          train,test = train_test_split(df, train_size=0.85, shuffle=False)

          data = train['SALES_OFFLINE']
          
          # Set frequency of time series to monthly
          data = data.asfreq('MS')
          
          # Define search ranges for p, d, q and seasonal P, D, Q
          p = range(0, 3)
          d = range(0, 2)
          q = range(0, 3)
          
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
          
          # trattiamo il covid come variabile esogena per ARIMA
          exog = pd.DataFrame({'holiday': np.zeros(len(train))}, 
                              index=train.index)
          # settiamo 1 come dummy variable per tenere traccia del covid
          exog.loc[(exog.index >= '2020-02') & (exog.index <= '2020-05'), 'holiday'] = 1
      
          # avviamo la stepwise per trovare i migliori parametri di seasonality
          stepwise_fit = auto_arima(train['SALES_OFFLINE'], 
                                    seasonal=True, 
                                    # m=12, 
                                    trace=True, 
                                    suppress_warnings=True,
                                    exogenous=exog)
          # mettiamo in best_season i migliori parametri che poi andremo ad utilizzare nel modello 
          best_seasonal = stepwise_fit.to_dict()['order']
          best_seasonal = (*best_seasonal, 12)
          
          
          # definiamo il modello da utilizzare sul train
          model = ARIMA(train['SALES_OFFLINE'], 
                        order=best_params ,
                        seasonal_order=best_seasonal ,
                        exog=exog)
          # fittiamo il modello
          model_fit = model.fit()
          
          # creaimo le esogene anche per il test
          exog_test = pd.DataFrame({'holiday': np.zeros(len(test))}, 
                              index=test.index)
          
          # avviamo il modello sul test set
          test['SALES_OFFLINE_forecast'] = model_fit.predict(start=test.index[0], 
                                                             end=test.index[-1], 
                                                             exog=exog_test, 
                                                             dynamic=True, 
                                                             typ='levels')
          #calcoliamo il mape di ARIMA
          MAPE_arima = mean_absolute_percentage_error(test['SALES_OFFLINE'], test['SALES_OFFLINE_forecast'])
          
          
          # creiamo la colonna date che andremo a rinominare
          create_features(df)
          create_features(future)
      
        # rinominiamo le colonne in ds e y perche prophet per funzionare vuole le colonne in questo modo
          df_prophet = df.rename(columns={'date':'ds',
                               'SALES_OFFLINE':'y'})

          future= future.rename(columns={'date':'ds',
                               'SALES_OFFLINE':'y'})
          
          #  splittiamo train e test set
          train,test = train_test_split(df_prophet, train_size=0.85, shuffle=False)
          
          #  otteniamo il nostro bennchmark che useremo per calcolare il mape del benchmark
          test['mean'] = train['y'].mean()     
      
          'GRID SEARCH CHANGEPOINT PRIOR SCALE'
          
          # definiamo la grid search per  ogni parametro
          grid_change_scale = {'changepoint_prior_scale':[0.05,0.1,0.5,1,3,5,8,10,12]}  
          grid_change_scale = ParameterGrid(grid_change_scale)
          
              
          for p in grid_change_scale:
              # definiamo il modello 
              model = Prophet(changepoint_prior_scale=p['changepoint_prior_scale'],holidays = lockdowns)
              'scegliere se fittare la grid sul train o su tutto il df'
              # fittiamo il modello
              model.fit(df_prophet)
              # effettuiamo la prediction
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              change_prior_parameter = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              change_prior_parameter = change_prior_parameter.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
              # sortiamo i mape in ordine ascendente
              change_prior_parameter = change_prior_parameter.sort_values(by=['MAPE'])
              change_prior_parameter = change_prior_parameter.reset_index(drop=True)
              # il parametro all' index 0 sarà il migliore dato che avra il mape più basso
              best_changepoint = change_prior_parameter['Parameters'][0]
              best_changepoint  = best_changepoint['changepoint_prior_scale']
          
            # qui cambiamo la grid search della yearly seasonality in base alla lunghezza della serie, siccome non ha senso avere 
            #  dei parametri alti per una serie corta, ciò rallenterebbe solo l'esecuzione del codice
          if len(df_prophet) < 15:
              grid_yearly_seasonality = {'yearly_seasonality':[8,10,12,'auto']}
              
          elif len(df_prophet) < 24 and len(df_prophet) >  15 :
              grid_yearly_seasonality   = {'yearly_seasonality':[8,10,12,15,20,'auto']}
              
          elif len(df_prophet) < 36 and len(df_prophet) >  24 :
              grid_yearly_seasonality   = {'yearly_seasonality':[8,10,12,15,20,21,25,27,30,'auto']}
          
          elif len(df_prophet) < 48 and len(df_prophet) >  30 :
              grid_yearly_seasonality   = {'yearly_seasonality':[8,10,12,15,20,21,25,27,30,35,'auto']}
              
          elif len(df_prophet) > 48 :
               grid_yearly_seasonality   = {'yearly_seasonality':[8,10,12,15,20,21,25,27,30,35,37,40,'auto']}
              
             
            # il processo è lo stesso per ogni parametro, in totale sono 7
          'GRID SEARCH  YEARLY SEASONALITY'
          grid_yearly_seasonality = ParameterGrid(grid_yearly_seasonality)   
          for p in grid_yearly_seasonality:
              model = Prophet(yearly_seasonality=p['yearly_seasonality'],holidays= lockdowns, changepoint_prior_scale=best_changepoint)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              yearly_season_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              yearly_season_parameters = yearly_season_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)  
              yearly_season_parameters = yearly_season_parameters.sort_values(by=['MAPE'])
              yearly_season_parameters = yearly_season_parameters.reset_index(drop=True)
              best_yearly_seasonality= yearly_season_parameters['Parameters'][0]
              best_yearly_seasonality  = best_yearly_seasonality['yearly_seasonality']   
           
          'GRID SEARCH  N_CHANGEPOINTS'
          if len(df_prophet) < 86:
              grid_n_changepoints = {'n_changepoints':[8,10,12,20]}
          else: 
              grid_n_changepoints = {'n_changepoints':[8,10,12,20,25,30,40]}
              
          grid_n_changepoints = ParameterGrid(grid_n_changepoints) 
          for p in grid_n_changepoints:
              
              model = Prophet(n_changepoints=p['n_changepoints'], changepoint_prior_scale=best_changepoint,yearly_seasonality=best_yearly_seasonality)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              n_changepoints_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              n_changepoints_parameters = n_changepoints_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
              n_changepoints_parameters = n_changepoints_parameters.sort_values(by=['MAPE'])
              n_changepoints_parameters = n_changepoints_parameters.reset_index(drop=True)
              best_n_changepoints = n_changepoints_parameters['Parameters'][0]
              best_n_changepoints  = best_n_changepoints['n_changepoints']
                  
          'GRID SEARCH SEASONALITY PRIOR SCALE'
          grid_season_prior = {'seasonality_prior_scale':[0.05,0.1,0.4,1,2,]}
          grid_season_prior = ParameterGrid(grid_season_prior)
          for p in grid_season_prior:
              
              model = Prophet(seasonality_prior_scale=p['seasonality_prior_scale'],holidays= lockdowns,changepoint_prior_scale=best_changepoint
                              ,yearly_seasonality=best_yearly_seasonality,n_changepoints = best_n_changepoints)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              season_prior_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              season_prior_parameters = season_prior_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)     
              season_prior_parameters = season_prior_parameters.sort_values(by=['MAPE'])
              season_prior_parameters = season_prior_parameters.reset_index(drop=True)
              best_season_prior = season_prior_parameters['Parameters'][0]
              best_season_prior  = best_season_prior['seasonality_prior_scale']
             
          'GRID SEARCH  HOLIDAYS PRIOR SCALE' 
          grid_holi_prior = {'holidays_prior_scale':[0.05,0.1,0.5,1,]}
          grid_holi_prior = ParameterGrid(grid_holi_prior)
          for p in grid_holi_prior:
              
              model = Prophet(holidays_prior_scale=p['holidays_prior_scale'],holidays = lockdowns,changepoint_prior_scale = best_changepoint, 
                              yearly_seasonality=best_yearly_seasonality,n_changepoints = best_n_changepoints,
                              seasonality_prior_scale= best_season_prior)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              holy_prior_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              holy_prior_parameters = holy_prior_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)      
              holy_prior_parameters = holy_prior_parameters.sort_values(by=['MAPE'])
              holy_prior_parameters = holy_prior_parameters.reset_index(drop=True)
              best_holy_prior = holy_prior_parameters['Parameters'][0]
              best_holy_prior  = best_holy_prior['holidays_prior_scale']
           
          'GRID SEARCH GROWTH'
          growth_grid = {'growth':['linear','flat']}
          growth_grid = ParameterGrid(growth_grid)
          for p in growth_grid:
              
              model = Prophet(growth=p['growth'],holidays = lockdowns,changepoint_prior_scale = best_changepoint, 
                              yearly_seasonality=best_yearly_seasonality,n_changepoints = best_n_changepoints,
                              seasonality_prior_scale= best_season_prior,holidays_prior_scale=best_holy_prior)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              growth_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              growth_parameters = growth_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
              growth_parameters = growth_parameters.sort_values(by=['MAPE'])
              growth_parameters = growth_parameters.reset_index(drop=True)
              best_growth = growth_parameters['Parameters'][0]
              best_growth  = best_growth['growth']
             
              
              
          # trainiamo il modello finale
          model= Prophet(yearly_seasonality=best_yearly_seasonality ,holidays=lockdowns,changepoint_prior_scale=best_changepoint
                          ,n_changepoints = best_n_changepoints,seasonality_prior_scale=best_season_prior,
                          holidays_prior_scale=best_holy_prior,growth = best_growth)
          
          # fit
          model.fit(df_prophet)
          # predict
          prophet_pred_on_test = model.predict(test)     
          #mape della modello
          MAPE_prophet = mean_absolute_percentage_error(test['y'], prophet_pred_on_test['yhat'])   
          #mape della media
          MAPE_Mean = mean_absolute_percentage_error(test['y'], test['mean'])

          # ora che abbiamo testato sia ARIMA che Prophet sul test set, vediamo quale
          # modello ha raggiunto il mape minore e avviamo la predizione futura con quel modello
          # se il mape di ARIMA è il più basso usa il modello ARIMA per il forecast
          if MAPE_arima < MAPE_prophet and MAPE_arima < MAPE_Mean and MAPE_arima <1:
               
              # trattiamo il covid come variabile esogena per ARIMA
              exog_df = pd.DataFrame({'holiday': np.zeros(len(df))}, 
                              index=df.index)   
              exog_df.loc[(exog_df.index >= '2020-02') & (exog_df.index <= '2020-05'), 'holiday'] = 1
              
              # definiamo il modello
              arima_future = ARIMA(df['SALES_OFFLINE'], 
                            order=best_params ,
                            seasonal_order=best_seasonal ,
                            exog=exog_df)
              
              print(len(exog_df))
              # fittiamo il modello
              arima_future = arima_future.fit()
              
              exog_future = pd.DataFrame({'holiday': np.zeros(len(future))}, 
                              index=future.index)   
              # Make predictions
              ARIMA_prediction = arima_future.predict(start=future.index[0],
                                                   end=future.index[-1], 
                                                   exog= exog_future, 
                                                   dynamic=True, 
                                                   typ='levels')
              
              #trasformiamolo in dataframe
              ARIMA_prediction = pd.DataFrame(ARIMA_prediction)              
              # ARIMA_prediction = ARIMA_prediction.set_index('YEARMONTH', inplace=True)
              ARIMA_prediction['YEARMONTH'] = future.index
              # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
              ARIMA_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0]
              # rinominamo le colonne
              ARIMA_prediction = ARIMA_prediction.rename(columns={'predicted_mean':'PREDICTED_SALES_OFFLINE'})
              # creiamo una colonna scrivendo il modello utilizzato
              ARIMA_prediction['MODEL'] = 'ARIMA'
              # storiamo tutti i mape dei modelli per un confronto
              ARIMA_prediction['MAPE_ARIMA'] = MAPE_arima
              ARIMA_prediction['MAPE_PROPHET'] = MAPE_prophet
              ARIMA_prediction['MAPE_MEAN'] = MAPE_Mean
              
              print('using arima')
              return ARIMA_prediction
          
          #  se il mape di PROPHET è il più basso usa il modello PROPHET per il forecast    
          elif MAPE_prophet < MAPE_arima and MAPE_prophet < MAPE_Mean and MAPE_prophet <1:
              
              # definiamo il modello
              final_model = Prophet(yearly_seasonality=best_yearly_seasonality ,holidays=lockdowns,changepoint_prior_scale=best_changepoint
                              ,n_changepoints = best_n_changepoints,seasonality_prior_scale=best_season_prior,
                              holidays_prior_scale=best_holy_prior,growth = best_growth)
              # fit the model
              final_model = final_model.fit(df_prophet)
              
              # predict future
              prophet_prediction = final_model.predict(future)
              
              # la linea sottostante serve a creare un df con solo la variabile yhat (valore predetto) 
              # dato che prophet con la funzione predict crea un df con svariate variabili ma
              
              #storiamo le predizioni in una nuova variabile, in 'yhat' osno contenute le predizioni
              prophet_prediction = prophet_prediction['yhat'] 
              #trasformiamolo in dataframe
              prophet_prediction = pd.DataFrame(prophet_prediction)
              # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
              prophet_prediction['YEARMONTH'] = future.index
              # settiamolo  yearmonth come index
              # prophet_prediction = prophet_prediction.set_index('YEARMONTH', inplace=True)
              # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
              prophet_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[1]
              # rinominamo le colonne
              prophet_prediction = prophet_prediction.rename(columns={'yhat':'PREDICTED_SALES_OFFLINE'})
              # creiamo una colonna scrivendo il modello utilizzato
              prophet_prediction['MODEL'] = 'PROPHET'
              # storiamo tutti i mape dei modelli per un confronto
              prophet_prediction['MAPE_ARIMA'] = MAPE_arima
              prophet_prediction['MAPE_PROPHET'] = MAPE_prophet
              prophet_prediction['MAPE_MEAN'] = MAPE_Mean
              #la funzione ci da la variabile prediction che è un dataframe che contiene i valori predetti
              #con seguenti colonne ['yhat' ,'YEARMONTH','BRAND', 'PRODUCT_GROUP', 'SECTOR'] 
              #yhat sarebbe il valore predetto
    
              print('using prophet')
              return prophet_prediction
          
              
          #  se il mape minore è il mape della media usiamo il metodo naive
          elif MAPE_Mean < MAPE_arima and MAPE_Mean < MAPE_prophet:
              
              # usando il metodo naive: usiamo gli ultimi 12 mesi come predizione
              future_prediction = df['SALES_OFFLINE'][-12:]
              # trasformiamolo in un dataframe
              future_prediction = pd.DataFrame(future_prediction)
              # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
              future_prediction['YEARMONTH'] = future.index
              # settiamo  yearmonth come index
              # future_prediction = future_prediction.set_index('YEARMONTH', inplace=True)
              # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
              future_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0] 
              # rinominamo le colonne
              future_prediction = future_prediction.rename(columns={'SALES_OFFLINE': 'PREDICTED_SALES_OFFLINE'})
              # creiamo una colonna scrivendo il modello utilizzato
              future_prediction['MODEL'] = 'NAIVE'
              # storiamo tutti i mape dei modelli per un confronto
              future_prediction['MAPE_ARIMA'] = MAPE_arima
              future_prediction['MAPE_PROPHET'] = MAPE_prophet
              future_prediction['MAPE_MEAN'] = MAPE_Mean
              print('using naive')
              return future_prediction
          
          # se tutti i mape sono maggiori o uguali a 1 (100% di errore sul test set)
          # utilizziamo direttamente il metodo naive
          elif MAPE_Mean >= 1 and MAPE_arima >= 1 and MAPE_prophet >= 1:
              # usando il metodo naive: usiamo gli ultimi 12 mesi come predizione
              future_prediction = df['SALES_OFFLINE'][-12:]
              # trasformiamolo in un dataframe
              future_prediction = pd.DataFrame(future_prediction)
              # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
              future_prediction['YEARMONTH'] = future.index
              # settiamo  yearmonth come index
              # future_prediction = future_prediction.set_index('YEARMONTH', inplace=True)
              # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
              future_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0] 
              # rinominamo le colonne
              future_prediction = future_prediction.rename(columns={'SALES_OFFLINE': 'PREDICTED_SALES_OFFLINE'})
              # creiamo una colonna scrivendo il modello utilizzato
              future_prediction['MODEL'] = 'NAIVE'
              # storiamo tutti i mape dei modelli per un confronto
              future_prediction['MAPE_ARIMA'] = MAPE_arima
              future_prediction['MAPE_PROPHET'] = MAPE_prophet
              future_prediction['MAPE_MEAN'] = MAPE_Mean
              print('using naive')
              return future_prediction
          
    # se la percentuale di zeri all'interno della serie  o negli ultimi 12 mesi
    # è maggiore del 50% usiamo il metodo naive per predire il futuro
      else:
          # usando il metodo naive: usiamo gli ultimi 12 mesi come predizione
          future_prediction = df['SALES_OFFLINE'][-12:]
          # trasformiamolo in un dataframe
          future_prediction = pd.DataFrame(future_prediction)
          # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
          future_prediction['YEARMONTH'] = future.index
          # settiamo  yearmonth come index
          # future_prediction = future_prediction.set_index('YEARMONTH', inplace=True)
          # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
          future_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0] 
          # rinominamo le colonne
          future_prediction = future_prediction.rename(columns={'SALES_OFFLINE': 'PREDICTED_SALES_OFFLINE'})
          # creiamo una colonna scrivendo il modello utilizzato
          future_prediction['MODEL'] = 'NAIVE'
          # storiamo tutti i mape dei modelli per un confronto
          future_prediction['MAPE_ARIMA'] = 0
          future_prediction['MAPE_PROPHET'] = 0
          future_prediction['MAPE_MEAN'] = 0
          print('using naive')
          
          return future_prediction
      

# creiamo un df vuoto dove andremo a storare tutte le predizioni 
predictions_df_offline= pd.DataFrame() 
 
 # Loop through each dataframe in dict_df_slices and apply the models
for name,group in dict_model.items():
    print(f"Applying Prophet model to {name} - {group['PRODUCT_GROUP'].iloc[0]} - {group['BRAND'].iloc[0]}") 
    # storiamo le prediction in un df chiamato prediction
    prediction= apply_prophet_arima_model(group,future)
    
    # ogni iterazione appendiamo la nuova predizione a quelle gia fatte
    predictions_df_offline= pd.concat([predictions_df_offline,prediction])





'Andremo a fare la stessa cosa per le SALES_ONLNE'

# creiamo un df vuoto con i mesi da 03-22 a 02-23 che sono i mesi che andremo a predirre
'create future'
# create a list of months from February 2022 to February 2023
months = pd.date_range(start='2022-03-01', end='2023-02-01', freq='MS')
# create a DataFrame with the months as the index and an empty column
future_online = pd.DataFrame(index=months, columns=['SALES_ONLINE'])

#questa funzione serve a creare una colonna con la data 
def create_features(df):
    df['date'] = df.index
    
'ONLINE'
def apply_prophet_arima_model_online(df,future):
    
      #  settiamo la colonna yearmonth come index
      df.set_index('YEARMONTH', inplace=True)

      # calcola la percentuale di valori 0 nella colonna 'SALES_ONLINE'
      zero_percentage = sum(df['SALES_ONLINE'] == 0) / len(df)
      
      # storiamo gli ultimi 12 mesi del df in una variabile
      last_12_sales_online = df['SALES_ONLINE'][-12:]
      
      # calcola la percentuale di valori 0 delle vendite negli ultimi 12 mesi
      zero_percentage_12 = sum(last_12_sales_online == 0) / len(last_12_sales_online)
      

    # verifica se la percentuale di zeri è minore al 50%, se lo è avvia le predizioni con i modelli
      if zero_percentage < 0.5 or zero_percentage_12 < 0.5:
          
          'ARIMA ON TEST'
          # Load data   
          freq = infer_freq(df.index)
          
          # interpolate the missing dates
          if freq is None:
              start_date = df.index[0]
              end_date = df.index[-1]
              idx = pd.date_range(start=start_date, end=end_date, freq='MS')
              df = df.reindex(idx, fill_value=None)
              df = df.interpolate()

          # train e test split
          train,test = train_test_split(df, train_size=0.85, shuffle=False)
         
          data = train['SALES_ONLINE']
          
          # Set frequency of time series to monthly
          data = data.asfreq('MS')
          
          # Define search ranges for p, d, q and seasonal P, D, Q
          p = range(0, 3)
          d = range(0, 2)
          q = range(0, 3)
          
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
          
          # avviamo la stepwise per trovare i migliori parametri di seasonality
          stepwise_fit = auto_arima(train['SALES_ONLINE'], 
                                    seasonal=True, 
                                    # m=12, 
                                    trace=True, 
                                    suppress_warnings=True,
                                   )
          # mettiamo in best_season i migliori parametri che poi andremo ad utilizzare nel modello 
          best_seasonal = stepwise_fit.to_dict()['order']
          best_seasonal = (*best_seasonal, 12)
          
          
          # definiamo il modello da utilizzare sul train
          model = ARIMA(train['SALES_ONLINE'], 
                        order=best_params ,
                        seasonal_order=best_seasonal ,
                        )
          # fittiamo il modello
          model_fit = model.fit()
          
          # avviamo il modello sul test set
          test['SALES_ONLINE_forecast'] = model_fit.predict(start=test.index[0], 
                                                             end=test.index[-1],                                                             
                                                             dynamic=True, 
                                                             typ='levels')
          #calcoliamo il mape di ARIMA
          MAPE_arima = mean_absolute_percentage_error(test['SALES_ONLINE'], test['SALES_ONLINE_forecast'])
          
          
          # create the date column that we are going to rename
          create_features(df)
          create_features(future)
      
        # rinominiamo le colonne in ds e y perche prophet per funzionare vuole le colonne in questo modo
          df_prophet = df.rename(columns={'date':'ds',
                               'SALES_ONLINE':'y'})

          future= future.rename(columns={'date':'ds',
                               'SALES_ONLINE':'y'})
          
          #  splittiamo train e test set
          train,test = train_test_split(df_prophet, train_size=0.85, shuffle=False)
          
          #  otteniamo il nostro bennchmark che useremo per calcolare il mape del benchmark
          test['mean'] = train['y'].mean()     
      
          'GRID SEARCH CHANGEPOINT PRIOR SCALE'
          
          # definiamo i valori per la grid search di ogni parametro
          grid_change_scale = {'changepoint_prior_scale':[0.05,0.1,0.5,1,3,5,8,10,12]}  
          grid_change_scale = ParameterGrid(grid_change_scale)
          
              
          for p in grid_change_scale:
              # definiamo il modello 
              model = Prophet(changepoint_prior_scale=p['changepoint_prior_scale'])
              'scegliere se fittare la grid sul train o su tutto il df'
              # fittiamo il modello
              model.fit(df_prophet)
              # effettuiamo la prediction
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              change_prior_parameter = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              change_prior_parameter = change_prior_parameter.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
              # sortiamo i mape in ordine ascendente
              change_prior_parameter = change_prior_parameter.sort_values(by=['MAPE'])
              change_prior_parameter = change_prior_parameter.reset_index(drop=True)
              # il parametro all' index 0 sarà il migliore dato che avra il mape più basso
              best_changepoint = change_prior_parameter['Parameters'][0]
              best_changepoint  = best_changepoint['changepoint_prior_scale']
          
            # qui cambiamo la grid search della yearly seasonality in base alla lunghezza della serie, siccome non ha senso avere 
            #  dei parametri alti per una serie corta, ciò rallenterebbe solo l'esecuzione del codice
          if len(df_prophet) < 15:
              grid_yearly_seasonality = {'yearly_seasonality':[8,10,12,'auto']}
              
          elif len(df_prophet) < 24 and len(df_prophet) >  15 :
              grid_yearly_seasonality   = {'yearly_seasonality':[8,10,12,15,20,'auto']}
              
          elif len(df_prophet) < 36 and len(df_prophet) >  24 :
              grid_yearly_seasonality   = {'yearly_seasonality':[8,10,12,15,20,21,25,27,30,'auto']}
          
          elif len(df_prophet) < 48 and len(df_prophet) >  30 :
              grid_yearly_seasonality   = {'yearly_seasonality':[8,10,12,15,20,21,25,27,30,35,'auto']}
              
          elif len(df_prophet) > 48 :
               grid_yearly_seasonality   = {'yearly_seasonality':[8,10,12,15,20,21,25,27,30,35,37,40,'auto']}
              
             
            # il processo è lo stesso per ogni parametro, in totale sono 7
          'GRID SEARCH  YEARLY SEASONALITY'
          grid_yearly_seasonality = ParameterGrid(grid_yearly_seasonality)   
          for p in grid_yearly_seasonality:
              model = Prophet(yearly_seasonality=p['yearly_seasonality'],changepoint_prior_scale=best_changepoint)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              yearly_season_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              yearly_season_parameters = yearly_season_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)  
              yearly_season_parameters = yearly_season_parameters.sort_values(by=['MAPE'])
              yearly_season_parameters = yearly_season_parameters.reset_index(drop=True)
              best_yearly_seasonality= yearly_season_parameters['Parameters'][0]
              best_yearly_seasonality  = best_yearly_seasonality['yearly_seasonality']   
           
          'GRID SEARCH  N_CHANGEPOINTS'
          if len(df_prophet) < 86:
              grid_n_changepoints = {'n_changepoints':[8,10,12,20]}
          else: 
              grid_n_changepoints = {'n_changepoints':[8,10,12,20,25,30,40]}
              
          grid_n_changepoints = ParameterGrid(grid_n_changepoints) 
          for p in grid_n_changepoints:
              
              model = Prophet(n_changepoints=p['n_changepoints'], changepoint_prior_scale=best_changepoint,yearly_seasonality=best_yearly_seasonality)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              n_changepoints_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              n_changepoints_parameters = n_changepoints_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
              n_changepoints_parameters = n_changepoints_parameters.sort_values(by=['MAPE'])
              n_changepoints_parameters = n_changepoints_parameters.reset_index(drop=True)
              best_n_changepoints = n_changepoints_parameters['Parameters'][0]
              best_n_changepoints  = best_n_changepoints['n_changepoints']
                  
          'GRID SEARCH SEASONALITY PRIOR SCALE'
          grid_season_prior = {'seasonality_prior_scale':[0.05,0.1,0.4,1,2,]}
          grid_season_prior = ParameterGrid(grid_season_prior)
          for p in grid_season_prior:
              
              model = Prophet(seasonality_prior_scale=p['seasonality_prior_scale'],changepoint_prior_scale=best_changepoint
                              ,yearly_seasonality=best_yearly_seasonality,n_changepoints = best_n_changepoints)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              season_prior_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              season_prior_parameters = season_prior_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)     
              season_prior_parameters = season_prior_parameters.sort_values(by=['MAPE'])
              season_prior_parameters = season_prior_parameters.reset_index(drop=True)
              best_season_prior = season_prior_parameters['Parameters'][0]
              best_season_prior  = best_season_prior['seasonality_prior_scale']
             
          'GRID SEARCH  HOLIDAYS PRIOR SCALE' 
          grid_holi_prior = {'holidays_prior_scale':[0.05,0.1,0.5,1,]}
          grid_holi_prior = ParameterGrid(grid_holi_prior)
          for p in grid_holi_prior:
              
              model = Prophet(holidays_prior_scale=p['holidays_prior_scale'],changepoint_prior_scale = best_changepoint, 
                              yearly_seasonality=best_yearly_seasonality,n_changepoints = best_n_changepoints,
                              seasonality_prior_scale= best_season_prior)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              holy_prior_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              holy_prior_parameters = holy_prior_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)      
              holy_prior_parameters = holy_prior_parameters.sort_values(by=['MAPE'])
              holy_prior_parameters = holy_prior_parameters.reset_index(drop=True)
              best_holy_prior = holy_prior_parameters['Parameters'][0]
              best_holy_prior  = best_holy_prior['holidays_prior_scale']
           
          'GRID SEARCH GROWTH'
          growth_grid = {'growth':['linear','flat']}
          growth_grid = ParameterGrid(growth_grid)
          for p in growth_grid:
              
              model = Prophet(growth=p['growth'],changepoint_prior_scale = best_changepoint, 
                              yearly_seasonality=best_yearly_seasonality,n_changepoints = best_n_changepoints,
                              seasonality_prior_scale= best_season_prior,holidays_prior_scale=best_holy_prior)
              'scegliere se fittare la grid sul train o su tutto il df'
              model.fit(df_prophet)
              forecast = model.predict(test)
              #creiamo un nuovo df dove andremo a storare i mape e vediamo qual'è il param. migliore
              growth_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
              # calcoliamo il mape
              MAPE = mean_absolute_percentage_error(test['y'],forecast['yhat'])
              # appendiamo i mape nel df
              growth_parameters = growth_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
              growth_parameters = growth_parameters.sort_values(by=['MAPE'])
              growth_parameters = growth_parameters.reset_index(drop=True)
              best_growth = growth_parameters['Parameters'][0]
              best_growth  = best_growth['growth']
             
              
              
          # trainiamo il modello finale
          model= Prophet(yearly_seasonality=best_yearly_seasonality ,changepoint_prior_scale=best_changepoint
                          ,n_changepoints = best_n_changepoints,seasonality_prior_scale=best_season_prior,
                          holidays_prior_scale=best_holy_prior,growth = best_growth)
          
          # fit
          model.fit(df_prophet)
          # predict
          prophet_pred_on_test = model.predict(test)     
          #mape della modello
          MAPE_prophet = mean_absolute_percentage_error(test['y'], prophet_pred_on_test['yhat'])   
          #mape della media
          MAPE_Mean = mean_absolute_percentage_error(test['y'], test['mean'])
          
          
          #  se il mape di ARIMA è il più basso usa il modello ARIMA per il forecast
          if MAPE_arima < MAPE_prophet and MAPE_arima < MAPE_Mean and MAPE_arima <1:
              # definiamo il modello
              arima_future = ARIMA(df['SALES_ONLINE'], 
                            order=best_params ,
                            seasonal_order=best_seasonal )
              # fittiamo il modello
              arima_future = arima_future.fit()
               
              # Make predictions
              future['SALES_ONLINE_forecast'] = arima_future.predict(start=future.index[0],
                                                   end=future.index[-1], 
                                                   dynamic=True, 
                                                   typ='levels')
              
              #storiamo le predizioni in una nuova variabile
              ARIMA_prediction = future['SALES_ONLINE_forecast']
              #trasformiamolo in dataframe
              ARIMA_prediction = pd.DataFrame(ARIMA_prediction)
              # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
              ARIMA_prediction['YEARMONTH'] = future.index
              # settiamolo  yearmonth come index
              # ARIMA_prediction = ARIMA_prediction.set_index('YEARMONTH', inplace=True)
              # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
              ARIMA_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df_prophet[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0]
              # rinominamo le colonne
              ARIMA_prediction = ARIMA_prediction.rename(columns={'SALES_ONLINE_forecast':'PREDICTED_SALES_ONLINE'})
              # creiamo una colonna scrivendo il modello utilizzato
              ARIMA_prediction['MODEL'] = 'ARIMA'
              # storiamo tutti i mape dei modelli per un confronto
              ARIMA_prediction['MAPE_ARIMA'] = MAPE_arima
              ARIMA_prediction['MAPE_PROPHET'] = MAPE_prophet
              ARIMA_prediction['MAPE_MEAN'] = MAPE_Mean
              
              print('using arima')
              return ARIMA_prediction
          
          #  se il mape di PROPHET è il più basso usa il modello PROPHET per il forecast    
          elif MAPE_prophet < MAPE_arima and MAPE_prophet < MAPE_Mean and MAPE_prophet <1:
              
              # definiamo il modello
              final_model = Prophet(yearly_seasonality=best_yearly_seasonality ,changepoint_prior_scale=best_changepoint
                              ,n_changepoints = best_n_changepoints,seasonality_prior_scale=best_season_prior,
                              holidays_prior_scale=best_holy_prior,growth = best_growth)
              # fit the model
              final_model = final_model.fit(df_prophet)
              
              # predict future
              prophet_prediction = final_model.predict(future)
              
              # la linea sottostante serve a creare un df con solo la variabile yhat (valore predetto) 
              # dato che prophet con la funzione predict crea un df con svariate variabili ma
              
              #storiamo le predizioni in una nuova variabile, in 'yhat' osno contenute le predizioni
              prophet_prediction = prophet_prediction['yhat'] 
              #trasformiamolo in dataframe
              prophet_prediction = pd.DataFrame(prophet_prediction)
              # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
              prophet_prediction['YEARMONTH'] = future.index
              # settiamolo  yearmonth come index
              # prophet_prediction = prophet_prediction.set_index('YEARMONTH', inplace=True)
              # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
              prophet_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[1]
              # rinominamo le colonne
              prophet_prediction = prophet_prediction.rename(columns={'yhat':'PREDICTED_SALES_ONLINE'})
              # creiamo una colonna scrivendo il modello utilizzato
              prophet_prediction['MODEL'] = 'PROPHET'
              # storiamo tutti i mape dei modelli per un confronto
              prophet_prediction['MAPE_ARIMA'] = MAPE_arima
              prophet_prediction['MAPE_PROPHET'] = MAPE_prophet
              prophet_prediction['MAPE_MEAN'] = MAPE_Mean
              #la funzione ci da la variabile prediction che è un dataframe che contiene i valori predetti
              #con seguenti colonne ['yhat' ,'YEARMONTH','BRAND', 'PRODUCT_GROUP', 'SECTOR'] 
              #yhat sarebbe il valore predetto
    
              print('using prophet')
              return prophet_prediction
          
              
          #  se il mape minore è il mape della media usiamo il metodo naive
          elif MAPE_Mean < MAPE_arima and MAPE_Mean < MAPE_prophet:
              
              # usando il metodo naive: usiamo gli ultimi 12 mesi come predizione
              future_prediction = df['SALES_ONLINE'][-12:]
              # trasformiamolo in un dataframe
              future_prediction = pd.DataFrame(future_prediction)
              # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
              future_prediction['YEARMONTH'] = future.index
              # settiamo  yearmonth come index
              # future_prediction = future_prediction.set_index('YEARMONTH', inplace=True)
              # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
              future_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0] 
              # rinominamo le colonne
              future_prediction = future_prediction.rename(columns={'SALES_ONLINE': 'PREDICTED_SALES_ONLINE'})
              # creiamo una colonna scrivendo il modello utilizzato
              future_prediction['MODEL'] = 'NAIVE'
              # storiamo tutti i mape dei modelli per un confronto
              future_prediction['MAPE_ARIMA'] = MAPE_arima
              future_prediction['MAPE_PROPHET'] = MAPE_prophet
              future_prediction['MAPE_MEAN'] = MAPE_Mean
              print('using naive')
              return future_prediction
          
          # se tutti i mape sono maggiori o uguali a 1 (100% di errore sul test set)
          # utilizziamo direttamente il metodo naive
          elif MAPE_Mean > 1 and MAPE_arima > 1 and MAPE_prophet > 1:
              # usando il metodo naive: usiamo gli ultimi 12 mesi come predizione
              future_prediction = df['SALES_ONLINE'][-12:]
              # trasformiamolo in un dataframe
              future_prediction = pd.DataFrame(future_prediction)
              # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
              future_prediction['YEARMONTH'] = future.index
              # settiamo  yearmonth come index
              # future_prediction = future_prediction.set_index('YEARMONTH', inplace=True)
              # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
              future_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0] 
              # rinominamo le colonne
              future_prediction = future_prediction.rename(columns={'SALES_ONLINE': 'PREDICTED_SALES_ONLINE'})
              # creiamo una colonna scrivendo il modello utilizzato
              future_prediction['MODEL'] = 'NAIVE'
              # storiamo tutti i mape dei modelli per un confronto
              future_prediction['MAPE_ARIMA'] = MAPE_arima
              future_prediction['MAPE_PROPHET'] = MAPE_prophet
              future_prediction['MAPE_MEAN'] = MAPE_Mean
              print('using naive')
              return future_prediction
          
    # se la percentuale di zeri all'interno della serie  o negli ultimi 12 mesi
    # è maggiore del 50% usiamo il metodo naive per predire il futuro
        
      else:
          # usando il metodo naive: usiamo gli ultimi 12 mesi come predizione
          future_prediction = df['SALES_ONLINE'][-12:]
          # trasformiamolo in un dataframe
          future_prediction = pd.DataFrame(future_prediction)
          # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
          future_prediction['YEARMONTH'] = future.index
          # settiamo  yearmonth come index
          # future_prediction = future_prediction.set_index('YEARMONTH', inplace=True)
          # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
          future_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0] 
          # rinominamo le colonne
          future_prediction = future_prediction.rename(columns={'SALES_ONLINE': 'PREDICTED_SALES_ONLINE'})
          # creiamo una colonna scrivendo il modello utilizzato
          future_prediction['MODEL'] = 'NAIVE'
          # storiamo tutti i mape dei modelli per un confronto
          future_prediction['MAPE_ARIMA'] = 0
          future_prediction['MAPE_PROPHET'] = 0
          future_prediction['MAPE_MEAN'] = 0
          print('using naive')
          
          return future_prediction

# creiamo un df vuoto dove andremo a storare tutte le predizioni online
predictions_df_online= pd.DataFrame() 
 
 # Loop through each dataframe in dict_df_slices and apply the models
for name,group in dict_model.items():
    print(f"Applying Prophet model to {name} - {group['PRODUCT_GROUP'].iloc[0]} - {group['BRAND'].iloc[0]}") 
    # storiamo le prediction in un df chiamato prediction_online
    prediction_online= apply_prophet_arima_model_online(group,future)
    
    # ogni iterazione appendiamo la nuova predizione a quelle gia fatte
    predictions_df_online = pd.concat([predictions_df_online,prediction_online])

 #reset index
predictions_df_online = predictions_df_online.reset_index()
# droppiamo l'index
predictions_df_online = predictions_df_online.drop('index', axis = 1)
# settiamo la colonna yearmonth come index
predictions_df_online = predictions_df_online.set_index('YEARMONTH')

# definiamo una funzione per calcolare la media
def calcola_media(x):
    somma = sum(x)
    media = somma / len(x)
    return media

'MAPE OFFLINE'
# creiamo un dataframe con tutti i mape per le sales offline
df_mape_offline = pd.DataFrame({
    'PROPHET': predictions_df_offline['MAPE_PROPHET'],
    'ARIMA' : predictions_df_offline['MAPE_ARIMA'],
    'MEAN' : predictions_df_offline['MAPE_MEAN'],
    })

# droppiamo i mape = 0 dato che sono i naive e influirebbero sul calcolo del mape medio dei nostri modelli
df_mape_offline = df_mape_offline[df_mape_offline['MAPE_ARIMA'] != 0]
    
# leviamo tutti gli altri naive
df_mape_offline =df_mape_offline[df_mape_offline['MODEL'] != 'NAIVE']

# Create new column that concatenates brand, product_group and sector
df_mape_offline['group'] = df_mape_offline['BRAND'] + ' - ' + df_mape_offline['PRODUCT_GROUP'] + ' -' + df_mape_offline['SECTOR']

# Drop duplicates based on the new column
df_mape_offline = df_mape_offline.drop_duplicates(subset='group')

# Find column name with smallest value
df_mape_offline['min_column'] = df_mape_offline.idxmin(axis=1)
# creiamo una colonna con il mape minore tra i 3, ovvero quello del modello utilizzato
df_mape_offline['smallest_mape'] = df_mape_offline.min(axis=1)

# tramite la funzione value counts andiamo a vedere il quanti modelli Arima e Prophet
# sono stati utilizzati
df_mape_offline['min_column'].value_counts()
#calcoliamo l'errore medio aspettato sul totale delle nostre predizioni offline
calcola_media(df_mape_offline['smallest_mape'])


'MAPE ONLINE'
# creiamo un dataframe con tutti i mape per le sales online
df_mape_online = pd.DataFrame({
    'PROPHET': predictions_df_online['MAPE_PROPHET'],
    'ARIMA' : predictions_df_online['MAPE_ARIMA'],
    'MEAN' : predictions_df_online['MAPE_MEAN'],
    })

# droppiamo i mape = 0 dato che sono i naive e influirebbero sul calcolo del mape medio dei nostri modelli
df_mape_online = df_mape_online[df_mape_online['MAPE_ARIMA'] != 0]
    
# leviamo tutti gli altri naive
df_mape_online =df_mape_online[df_mape_online['MODEL'] != 'NAIVE']

# Create new column that concatenates brand, product_group and sector
df_mape_online['group'] = df_mape_online['BRAND'] + ' - ' + df_mape_online['PRODUCT_GROUP'] + ' -' + df_mape_online['SECTOR']

# Drop duplicates based on the new column
df_mape_online = df_mape_online.drop_duplicates(subset='group')

# Find column name with smallest value
df_mape_online['min_column'] = df_mape_online.idxmin(axis=1)
# creiamo una colonna con il mape minore tra i 3, ovvero quello del modello utilizzato
df_mape_online['smallest_mape'] = df_mape_online.min(axis=1)
#calcoliamo l'errore medio aspettato sul totale delle nostre predizioni
calcola_media(df_mape_online['smallest_mape'])

# tramite la funzione value counts andiamo a vedere il quanti modelli Arima e Prophet
# sono stati utilizzati
df_mape_online['min_column'].value_counts()


'MEAN FOR <12 MONTH SERIES'
# creiamo il dictionary contenente solo le serie che meno di 12 mesi che andremo a predire con la media
dict_naive = {}
for name, group in dict_df_slices.items():
    # Check if the dataframe has a valid row coun
    if len(group) < 12:
        dict_naive[name] = group
        
        

# create a DataFrame with the months as the index and an empty column
future_dates = pd.DataFrame(index=months, columns=['SALES_OFFLINE'])

# creiamo un df vuoto dove andremo a storare le predizioni
predictions_df_naive= pd.DataFrame() 

# definiamo la funzione da applicare in modo tale da usare la media per le predizioni
def naive(df,future_dates):
    # media sales online
    mean_online = df['SALES_ONLINE'].mean()
    # media sales offline
    mean_offline = df['SALES_OFFLINE'].mean()
    # creiamo un df dove usiamo la media per predire i prossimi 12 mesi
    future = pd.DataFrame({'mean_online': [mean_online]*12,'mean_offline': [mean_offline]*12 })
    # trasformiamolo in un dataframe
    future_prediction = pd.DataFrame(future)
    # creiamo la colonna yearmonth usando l'index di future, i mesi futuri da predirre
    future_prediction['YEARMONTH'] = future_dates.index
    # creiamo le colonne brand, product_group e sector utilizzando quelle del df che passiamo nella funzione
    future_prediction[['BRAND', 'PRODUCT_GROUP', 'SECTOR']] = df[['BRAND', 'PRODUCT_GROUP', 'SECTOR']].iloc[0] 
    # rinominamo le colonne
    future_prediction = future_prediction.rename(columns={'SALES_ONLINE': 'PREDICTED_SALES_ONLINE',' SALES_OFFLINE': 'PREDICTED_SALES_OFFLINE'})
    # creiamo una colonna scrivendo il modello utilizzato
    future_prediction['MODEL'] = 'NAIVE'
    # storiamo tutti i mape dei modelli per un confronto
    future_prediction['MAPE_ARIMA'] = 0
    future_prediction['MAPE_PROPHET'] = 0
    future_prediction['MAPE_MEAN'] = 0
    
    return future_prediction
    
 # Loop through each dataframe in dict_naive and apply the models
for name,group in dict_naive.items():
    
    prediction_naive = naive(group,future_dates)
    
    # ogni iterazione appendiamo la nuova predizione a quelle gia fatte
    predictions_df_naive = pd.concat([predictions_df_naive,prediction_naive])
    
#reset index
predictions_df_naive = predictions_df_naive.reset_index()
# droppiamo l'index
predictions_df_naive = predictions_df_naive.drop('index', axis = 1)
# settiamo la colonna yearmonth come index
predictions_df_naive = predictions_df_naive.set_index('YEARMONTH')

'!!!!!!apire come mergiare tutto siccome son presenti colonne differenti !!!!!!!'

#select only the column that we need for the csv
naive = predictions_df_naive[['PREDICTED_SALES_OFFLINE','PREDICTED_SALES_ONLINE', 'BRAND', 'PRODUCT_GROUP', 'SECTOR','YEARMONTH']]

offline = predictions_df_offline[['PREDICTED_SALES_OFFLINE', 'BRAND', 'PRODUCT_GROUP', 'SECTOR','YEARMONTH']]

online = [['PREDICTED_SALES_ONLINE', 'BRAND', 'PRODUCT_GROUP', 'SECTOR','YEARMONTH']]

# we have to merge this dataframe with the dataframe of the predicted values with the models
off_on_merged = pd.merge(offline,online, on=['BRAND', 'PRODUCT_GROUP', 'SECTOR', 'YEARMONTH'])

# store the df's in a list to merge the with concat
df_list = [off_on_merged,naive]

# merge the df's and obtain the final df
final_df = pd.concat(df_list)

# elimina valori negativi e li imputiamo a 0
final_df['PREDICTED_SALES_OFFLINE'] = final_df['PREDICTED_SALES_OFFLINE'].apply(lambda x: max(0, x))

# elimina valori negativi e li imputiamo a 0
final_df['PREDICTED_SALES_ONLINE'] = final_df['PREDICTED_SALES_ONLINE'].apply(lambda x: max(0, x))


final_df.to_csv('Sales_forecasting.csv')
























        
      
        
      
        