# Covid-19
To forecast the Covid-19 cases for next ‘n’ days using available time-series libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df=pd.read_csv("/content/Covid-2022.csv")
df.head()df=pd.read_csv("/content/Covid-2022.csv")
df.head()

#calculate the active cases
df['Active Cases'] = df['Confirmed Cases'] - df['Death'] - df['Cured/Discharged']
df.tail()

#grouping of the data using groupby
india = df.groupby('Date')['Confirmed Cases','Death','Cured/Discharged','Active Cases'].sum().reset_index()
india.head()

!pip install pystan
!pip install prophet

#import fbprophet library
from prophet import Prophet
#forecast for 7 upcoming days
confirmed=df.groupby('Date').sum()['Confirmed Cases'].reset_index()
cured=df.groupby('Date').sum()['Cured/Discharged'].reset_index()
deaths=df.groupby('Date').sum()['Death'].reset_index()
active=df.groupby('Date').sum()['Active Cases'].reset_index()
#rename the column of date with ds and confirmed cases with y
confirmed.columns=['ds','y']
confirmed['ds']=pd.to_datetime(confirmed['ds'])
confirmed.tail(29)
#building the model and running the algorithm
m=Prophet(interval_width=0.95)#Confidence level-model will try to provide the accuracy of 95%
m.fit(confirmed)
future=m.make_future_dataframe(periods=60)
future.tail(30)
#forecast the future with date,upper and lower limit of y
forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(30)
confirmed_plot=m.plot(forecast)
#deaths
deaths.columns=['ds','y']
deaths['ds']=pd.to_datetime(deaths['ds'])
#building the model and running the algorithm
m=Prophet(interval_width=0.95)#Confidence level-model will try to provide the accuracy of
m.fit(deaths)
future=m.make_future_dataframe(periods=60)
future.tail(60)
#forecast the future with date,upper and lower limit of y
forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(100)
death_plot=m.plot(forecast)
#cured
cured.columns=['ds','y']
cured['ds']=pd.to_datetime(cured['ds'])
#building the model and running the algorithm
m=Prophet(interval_width=0.95)#Confidence level-model will try to provide the accuracy of
m.fit(cured)
future=m.make_future_dataframe(periods=60)
future.tail(10)
#forecast the future with date,upper and lower limit of y
forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
cured_plot=m.plot(forecast)
Active_plot=m.plot(forecast)
import json 
import requests
fig = px.choropleth(
    df,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='Region',
    color='Active Cases',
    range_color=[1,2000],
    color_continuous_scale='earth'
    
)

fig.update_geos(fitbounds="locations", visible=False)

fig.show()
