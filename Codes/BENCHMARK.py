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

##################
################## apple smartphone
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


###############################
###############################
###############################

apple_smartphone_OFF_mean = apple_smartphone_OFF_test['SALES_OFFLINE'].copy()

mean = apple_smartphone_OFF_train['SALES_OFFLINE'].mean()

apple_smartphone_OFF_mean = pd.DataFrame(apple_smartphone_OFF_mean)

apple_smartphone_OFF_mean = apple_smartphone_OFF_mean.assign(SALES_OFFLINE_MEAN=mean)

apple_smartphone_OFF_mean = apple_smartphone_OFF_mean.drop('SALES_OFFLINE', axis=1)


# plottiamo
plt.plot(apple_smartphone_OFF_test)
plt.plot(apple_smartphone_OFF_mean)
plt.legend()
plt.show()


from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(apple_smartphone_OFF_test, apple_smartphone_OFF_mean)
print('MAPE:', mape)

















