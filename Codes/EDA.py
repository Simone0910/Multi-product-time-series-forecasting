
import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick


melt = pd.read_csv('//Users/domealbe/Desktop/DATA_SCIENCE/unieuro/luiss-businesscase2-market_data.csv', delimiter=';')
melt = melt.sort_values(by=["PRODUCT_GROUP", 'BRAND', "YEARMONTH"], ascending=[True, True, True])
melt= melt.sort_values(by=["YEARMONTH"], ascending=[True])
melt['YEARMONTH'] = pd.to_datetime(melt['YEARMONTH'], format='%Y%m')
# melt['YEARMONTH'] = melt['YEARMONTH'].dt.strftime('%Y-%m')

df_all = melt[['YEARMONTH','SALES_ONLINE', 'SALES_OFFLINE','SALES_TOTAL', 'BRAND','PRODUCT_GROUP','SECTOR']]
df_all = df_all.sort_values(by=['PRODUCT_GROUP','BRAND','YEARMONTH'])

gruppi = df_all.groupby(['BRAND', 'PRODUCT_GROUP'])
dict_df_slices = {name: group for name, group in gruppi}
# dalla size del dict creato notiamo che ci sono 3217 time series 


####################################
####################################
'EXPLORATORY DATA ANALYSIS'

# contiamo quante sono le time series che hanno almeno 20 mesi di dati
count_valid = 0

for key, value in dict_df_slices.items():
    if len(value) >= 20:
        count_valid += 1
print(f"Il numero di time series con almeno 12 mesi di dati è {count_valid} su un totale di {len(dict_df_slices)} time series.")


# Contatore delle time series con mesi mancanti
count_incomplete = 0

# Ciclo per verificare se ci sono time series incomplete
for key, value in dict_df_slices.items():
    # Verifica se ci sono date mancanti nella time series dall'inizio alla fine
    start_date = pd.to_datetime(value['YEARMONTH'].min())
    end_date = pd.to_datetime(value['YEARMONTH'].max())
    missing_dates = pd.date_range(start=start_date, end=end_date, freq='MS').difference(pd.to_datetime(value['YEARMONTH']))
    
    if len(missing_dates) > 0:
        count_incomplete += 1

print(f"Le time series con mesi mancanti sono {count_incomplete} su un totale di {len(dict_df_slices)} time series")



count_valid = 0
count_incomplete = 0

# Ciclo per verificare se ci sono time series incomplete
for key, value in dict_df_slices.items():
    if len(value) >= 20:
        # Verifica se ci sono date mancanti nella time series dall'inizio alla fine
        start_date = pd.to_datetime(value['YEARMONTH'].min())
        end_date = pd.to_datetime(value['YEARMONTH'].max())
        missing_dates = pd.date_range(start=start_date, end=end_date, freq='MS').difference(pd.to_datetime(value['YEARMONTH']))

        if len(missing_dates) > 0:
            count_incomplete += 1

        count_valid += 1

print(f"Le time series con mesi mancanti e almeno 12 mesi di dati sono {count_incomplete} su un totale di {count_valid} time series.")

#quindi abbiamo un totale di 764 time series che dovremo gestire in qualche modo per i valori mancanti

# questi valori mancanti li tratteremo in futuro in maniera specifica nei modelli.
# perchè ogni modello che applicheremo, avrà bisogno o meno di una gestione specifica
# dei missing values


########################### 
########################### plottiamo i product group per vedere sia quali sono i più venduti 
########################### sia quanta discrepanza c'è tra le vendite per gruppo di prodotto

lista_product_group = list(melt['PRODUCT_GROUP'].unique())
lista_brand = list(melt['BRAND'].unique())

df_prodotti = melt.groupby('PRODUCT_GROUP').agg({ 'SALES_TOTAL': 'sum', 'SALES_ONLINE': 'sum', 'SALES_OFFLINE': 'sum'})
df_prodotti = df_prodotti.sort_values(by=["SALES_TOTAL"], ascending=[False])


# Creazione del barplot
ax = df_prodotti[['SALES_ONLINE', 'SALES_OFFLINE']].plot(kind='bar', stacked=True, color=['blue', 'orange'], figsize=(50, 10))
ax.grid(axis='y', linestyle='-', alpha=0.2)

# Etichette degli assi e del titolo
ax.set_xlabel('PRODUCT GROUP')
ax.set_ylabel('SALES')
ax.set_title('ONLINE SALES AND OFFLINE SALES FOR EACH PRODUCT GROUP')

plt.show()

### essendo i product group ben 117 e avendo valori di vendite completamente diversi (alcuni in decine di miliardi, altri in migliaia)
### plottarli tutti insieme non ha senso, concentriamoci su alcuni subset


### top 5 product group più venduti

df_prodotti_top5 = df_prodotti.head(5)
# Creazione del barplot
ax = df_prodotti_top5[['SALES_ONLINE', 'SALES_OFFLINE']].plot(kind='bar', stacked=True, color=['blue', 'orange'], figsize=(20, 20))
ax.grid(axis='y', linestyle='-', alpha=0.5)
# Etichette degli assi e del titolo
ax.set_xlabel('PRODUCT GROUP', fontsize=35)
ax.set_ylabel('SALES', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=35)
plt.xticks(rotation=45)
ax.set_title('Online and Offline Sales of the 5 most sold Product groups', fontsize=40)
# Visualizzazione del plot
plt.show()

# top 20
df_prodotti_top20 = df_prodotti.head(20)
# Creazione del barplot
ax = df_prodotti_top20[['SALES_ONLINE', 'SALES_OFFLINE']].plot(kind='bar', stacked=True, color=['blue', 'orange'], figsize=(40, 10))
ax.grid(axis='y', linestyle='-', alpha=0.5)
# Etichette degli assi e del titolo
ax.set_xlabel('PRODUCT GROUP')
ax.set_ylabel('SALES')
ax.set_title('Online and Offline Sales of the 5 most sold Product groups')
# Visualizzazione del plot
plt.show()



# last 20 
df_prodotti_last = df_prodotti.iloc[-20:]
ax = df_prodotti_last[['SALES_ONLINE', 'SALES_OFFLINE']].plot(kind='bar', stacked=True, color=['blue', 'orange'], figsize=(40, 10))
ax.grid(axis='y', linestyle='-', alpha=0.5)
# Etichette degli assi e del titolo
ax.set_xlabel('PRODUCT GROUP')
ax.set_ylabel('SALES')
ax.set_title('Online and Offline Sales of the remaining Product Groups')
# Visualizzazione del plot
plt.show()



'''
abbiamo visto che "smartphone" è il product group più venduto.
ora vediamo qual è una time series di questo product group che rispecchi 
il trend generale di vendite sia per le online sia per le offline.
la vogliamo trovare per iniziare a studiare e applicare alcuni modelli e ne
cerchiamo una appunto che rispecchi per stagionalità il trend generale.
'''


# Creiamo un nuovo dizionario in cui le chiavi sono i prodotti e i valori sono le somme delle vendite di ogni prodotto
prodotti_somme_vendite = {}
for prodotto, vendite in dict_df_slices.items():
    prodotti_somme_vendite[prodotto] = vendite.sum()['SALES_OFFLINE']

# Ordiniamo il dizionario in base ai valori (ovvero le somme delle vendite) in ordine decrescente e stampiamo i primi 5 elementi
for prodotto, somma_vendite in sorted(prodotti_somme_vendite.items(), key=lambda item: item[1], reverse=True)[:5]:
    print(prodotto)

### TOP 5 PRODUCT_GROUP E BRAND PIU' VENDUTI PER LE SALES OFFLINE
# 1     ('SAMSUNG', 'SMARTPHONES')
# 2     ('APPLE', 'SMARTPHONES')
# 3     ('HUAWEI', 'SMARTPHONES')
# 4     ('SAMSUNG', 'PTV/FLAT')
# 5     ('LG', 'PTV/FLAT')



# rifaccio la stessa cosa per le vendite online
prodotti_somme_vendite = {}
for prodotto, vendite in dict_df_slices.items():
    prodotti_somme_vendite[prodotto] = vendite.sum()['SALES_ONLINE']

# Ordiniamo il dizionario in base ai valori (ovvero le somme delle vendite) in ordine decrescente e stampiamo i primi 5 elementi
for prodotto, somma_vendite in sorted(prodotti_somme_vendite.items(), key=lambda item: item[1], reverse=True)[:5]:
    print(prodotto)

### TOP 5 PRODUCT_GROUP E BRAND PIU' VENDUTI PER LE SALES ONLINE
# 1     ('APPLE', 'SMARTPHONES')
# 2     ('SAMSUNG', 'SMARTPHONES')
# 3     ('SAMSUNG', 'PTV/FLAT')
# 4     ('HP', 'MOBILE COMPUTING')
# 5     ('LG', 'PTV/FLAT')


''' ora andiamo a vedere il trend di apple smartphone e 
compariamolo a quello delle vendite generali'''


apple_smartphone = dict_df_slices['APPLE', 'SMARTPHONES']
apple_smartphone.set_index('YEARMONTH', inplace=True)

apple_smartphone_OFF = apple_smartphone.filter(['SALES_OFFLINE', 'PRODUCT_GROUP', 'BRAND', 'SECTOR'])
apple_smartphone_OFF = apple_smartphone_OFF.drop(['PRODUCT_GROUP', 'BRAND', 'SECTOR'], axis = 1)
apple_smartphone_ON = apple_smartphone.filter(['SALES_ONLINE', 'PRODUCT_GROUP', 'BRAND', 'SECTOR'])
apple_smartphone_ON = apple_smartphone_ON.drop(['PRODUCT_GROUP', 'BRAND', 'SECTOR'], axis = 1)


df_grouped_offline = melt.groupby('YEARMONTH')['SALES_OFFLINE'].sum()
df_grouped_online = melt.groupby('YEARMONTH')['SALES_ONLINE'].sum()



# PLOT TREND GENERALE SALES OFFLINE
plt.xlim(['2015-01', '2022-03'])

f, ax = plt.subplots(figsize=(25, 5), dpi=300)
plt.title('TOTAL SALES OFFLINE TO SEE THE TREND', fontsize=15)

plt.plot(df_grouped_offline, label='SALES ONLINE TOTAL')

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
plt.xlim(['2014-12', '2022-03'])

plt.show()




# PLOT TREND APPLE SMARTPHONE SALES OFFLINE
plt.xlim(['2015-01', '2022-03'])

f, ax = plt.subplots(figsize=(25, 5), dpi=300)
plt.title('APPLE SMARTPHONE SALES OFFLINE TO SEE THE TREND', fontsize=15)

plt.plot(apple_smartphone_OFF['SALES_OFFLINE'], label='SALES OFFLINE')

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
plt.xlim(['2014-12', '2022-03'])

plt.show()





# PLOT TREND GENERALE SALES ONLINE
plt.xlim(['2015-01', '2022-03'])

f, ax = plt.subplots(figsize=(25, 5), dpi=300)
plt.title('TOTAL SALES ONLINE TO SEE THE TREND', fontsize=15)

plt.plot(df_grouped_online, label='SALES ONLINE TOTAL')

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
plt.xlim(['2014-12', '2022-03'])

plt.show()





# PLOT TREND APPLE SMARTPHONE SALES ONLINE
plt.xlim(['2015-01', '2022-03'])

f, ax = plt.subplots(figsize=(25, 5), dpi=300)
plt.title('APPLE SMARTPHONE SALES ONLINE TO SEE THE TREND', fontsize=15)

plt.plot(apple_smartphone_ON['SALES_ONLINE'], label='SALES ONLINE')

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
plt.xlim(['2014-12', '2022-03'])

plt.show()


################################
################################

'''
adesso, facciamo alcune analisi sui sectors e sui product group.
queste analisi saranno utili dopo, quando vorremmo trarre conclusioni di business
e capire i risultati economici del nostro studio.
'''

'''
per prima cosa ANALIZZIAMO I PRODUCT GROUP
e torniamo ai top 5 most sold product group 
'''

lista_product_group = list(melt['PRODUCT_GROUP'].unique()) # abbiamo 117 prod group
lista_brand = list(melt['BRAND'].unique()) # abbiamo 1201 brand

df_prodotti = melt.groupby('PRODUCT_GROUP').agg({ 'SALES_TOTAL': 'sum', 'SALES_ONLINE': 'sum', 'SALES_OFFLINE': 'sum'})
df_prodotti = df_prodotti.sort_values(by=["SALES_TOTAL"], ascending=[False])

df_prodotti_top5 = df_prodotti.head(5)
df_prodotti_top5.head(10)


# Set color values
orange_color = '#FF9A31'
blue_color = '#1e90ff'


# Create top 5 bar plot
df_prodotti_top5 = df_prodotti.head(5)
ax = df_prodotti_top5[['SALES_ONLINE']].plot(kind='bar', stacked=True, color=blue_color, figsize=(20, 20))
df_prodotti_top5[['SALES_OFFLINE']].plot(kind='bar', stacked=True, color=orange_color, figsize=(20, 20), bottom=df_prodotti_top5['SALES_ONLINE'], ax=ax)
ax.grid(axis='y', linestyle='-', alpha=0.5)
fig = ax.get_figure()
fig.set_dpi(500)

# Set axis labels and title
ax.set_xlabel('PRODUCT GROUP', fontsize=35, color='white')
ax.set_ylabel('SALES', fontsize=35, color='white')
ax.set_title('Online and Offline Sales of the 5 most sold Product groups', fontsize=40, color='white')
ax.set_facecolor((9/255, 29/255, 64/255))
ax.set_facecolor((0.035, 0.114, 0.251))

# Imposta i numeri interi sull'asse y
plt.ticklabel_format(useOffset=False, style='plain', axis='y')

# Set tick label sizes and colors
plt.xticks(fontsize=25, color='white')
plt.yticks(fontsize=25, color='white')

# Set legend font size and colors
plt.legend(fontsize=35, facecolor=(9/255, 29/255, 64/255), edgecolor='white')
fig = ax.get_figure()
fig.set_facecolor((9/255, 29/255, 64/255))


# Set bar colors
ax.get_children().set_color('#FF9A31')
ax.get_children().set_color('#1e90ff')

# Show plot
plt.show()



# CREIAMO UN DF CON LE VENDITE GENERALI DEI PRODUCT GROUP 
df_sum = df_all.groupby('PRODUCT_GROUP')['SALES_TOTAL'].sum().reset_index()
df_sum.head(5)

# ADESSO AI PRODUCT GROUP TOTALI APPENDIAMO LA PERCENTUALE DI VENDITE DEI SINGOLI PRODUCT GROUP
df_sum['percentuale_vendite'] = df_sum['SALES_TOTAL'] / df_sum['SALES_TOTAL'].sum()
df_sum = df_sum.sort_values('percentuale_vendite', ascending=False)
df_sum.head(5)


## PLOTTIAMO QUINDI LE PERCENTUALI DI VENDITA DEI TOP 5 PRODUCT GROUP
def percent(x, pos):
    return '{:.0f}%'.format(x*100)

formatter = mtick.FuncFormatter(percent)

# impostiamo i colori
bg_color = (0.035, 0.114, 0.251) # R9 G29 B64
bar_color = (1, 0.604, 0.192) # R255 G154 B49

# creiamo la figura e l'asse
fig, ax = plt.subplots()

# creiamo il grafico a barre
ax.bar(df_sum.head(5)['PRODUCT_GROUP'], df_sum.head(5)['percentuale_vendite'], color=bar_color)

# impostiamo il titolo e le etichette degli assi
ax.set_title('PERCENTAGE OF TOTAL SALES PER PRODUCT_GROUP', color='white')
ax.set_xlabel('PRODUCT_GROUP', fontsize=12, color='white')
ax.set_ylabel('% of TOTAL SALES', fontsize=12, color='white')

# impostiamo lo sfondo
fig.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# impostiamo i colori delle etichette degli assi
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# impostiamo il formato delle etichette sull'asse y
ax.yaxis.set_major_formatter(formatter)

# impostiamo la rotazione delle etichette sull'asse x
ax.set_xticklabels(df_sum.head(5)['PRODUCT_GROUP'], rotation=90, color='white')

# impostiamo la griglia
ax.grid(axis='y', alpha=0.2)

# visualizziamo il grafico
plt.show()

'''
ABBIAMO CAPITO DUNQUE LEGGENDO LE PERCENTUALI CHE SOLO QUESTI 5 PRODUCT GROUP 
RAPPRESENTANO INSIEME IL 58% DELLE VENDITE TOTALI
'''
##########

'ora spostiamoci ai SECTORS'

lista_sector = list(melt['SECTOR'].unique()) # abbiamo 9 settori

df_sector = melt.groupby('SECTOR').agg({ 'SALES_TOTAL': 'sum', 'SALES_ONLINE': 'sum', 'SALES_OFFLINE': 'sum'})
df_sector = df_sector.sort_values(by=["SALES_TOTAL"], ascending=[False])

df_sector = df_sector.head(5)
df_sector


# Set color values
orange_color = '#FF9A31'
blue_color = '#1e90ff'


# Create top 5 bar plot

ax = df_sector[['SALES_ONLINE']].plot(kind='bar', stacked=True, color=blue_color, figsize=(20, 20))
df_sector[['SALES_OFFLINE']].plot(kind='bar', stacked=True, color=orange_color, figsize=(20, 20), bottom=df_sector['SALES_ONLINE'], ax=ax)
ax.grid(axis='y', linestyle='-', alpha=0.5)
fig = ax.get_figure()
fig.set_dpi(500)

# Set axis labels and title
ax.set_xlabel('PRODUCT GROUP', fontsize=35, color='white')
ax.set_ylabel('SALES', fontsize=35, color='white')
ax.set_title('Online and Offline Sales of the 5 most sold Product groups', fontsize=40, color='white')
ax.set_facecolor((9/255, 29/255, 64/255))
ax.set_facecolor((0.035, 0.114, 0.251))

# Imposta i numeri interi sull'asse y
plt.ticklabel_format(useOffset=False, style='plain', axis='y')

# Set tick label sizes and colors
plt.xticks(fontsize=25, color='white')
plt.yticks(fontsize=25, color='white')

# Set legend font size and colors
plt.legend(fontsize=35, facecolor=(9/255, 29/255, 64/255), edgecolor='white')
fig = ax.get_figure()
fig.set_facecolor((9/255, 29/255, 64/255))


# Set bar colors
ax.get_children().set_color('#FF9A31')
ax.get_children().set_color('#1e90ff')

# Show plot
plt.show()






# CREIAMO UN DF CON LE VENDITE GENERALI DEI PRODUCT GROUP 
df_sec = df_all.groupby('SECTOR')['SALES_TOTAL'].sum().reset_index()
df_sec.head(10)

# ADESSO AI PRODUCT GROUP TOTALI APPENDIAMO LA PERCENTUALE DI VENDITE DEI SINGOLI PRODUCT GROUP
df_sec['percentuale_vendite'] = df_sec['SALES_TOTAL'] / df_sec['SALES_TOTAL'].sum()
df_sec = df_sec.sort_values('percentuale_vendite', ascending=False)
df_sec.head(10)



## PLOTTIAMO QUINDI LE PERCENTUALI DI VENDITA DEI TOP 5 PRODUCT GROUP
def percent(x, pos):
    return '{:.0f}%'.format(x*100)

formatter = mtick.FuncFormatter(percent)

# impostiamo i colori
bg_color = (0.035, 0.114, 0.251) # R9 G29 B64
bar_color = (1, 0.604, 0.192) # R255 G154 B49

# creiamo la figura e l'asse
fig, ax = plt.subplots()

# creiamo il grafico a barre
ax.bar(df_sec.head(10)['SECTOR'], df_sec.head(10)['percentuale_vendite'], color=bar_color)

# impostiamo il titolo e le etichette degli assi
ax.set_title('PERCENTAGE OF TOTAL SALES PER SECTOR', color='white')
ax.set_xlabel('SECTORS', fontsize=12, color='white')
ax.set_ylabel('% of TOTAL SALES', fontsize=12, color='white')

# impostiamo lo sfondo
fig.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# impostiamo i colori delle etichette degli assi
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# impostiamo il formato delle etichette sull'asse y
ax.yaxis.set_major_formatter(formatter)

# impostiamo la rotazione delle etichette sull'asse x
ax.set_xticklabels(df_sec.head(5)['SECTOR'], rotation=90, color='white')

# impostiamo la griglia
ax.grid(axis='y', alpha=0.2)

# visualizziamo il grafico
plt.show()


### Da ciò vediamo che il market share dei settori è:
# TELECOM 36.1%
# INFORMATION TECH/OFFICE EQUIPMENT 20.6%
# MAJOR DOMESTIC APPLIANCES 14.5%
# CONSUMER ELECTRONICS 14.2%
# PICCOLO ELETTRODOMESTICO   9.6%
# PHOTO   2.1%
# HOME COMFORT   1.9%
# MEDIA STORAGE   0.7%
# SPORT   0.2%

'''
The top 5 sectors cover 95% of total sales, 
while the top 7 cover 99%.

this maybe will influence our forecasting in which from the macro proint of view
we will treat the last two with other methods being not significant
'''








