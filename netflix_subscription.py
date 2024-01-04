import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objs as graph_objs
import plotly.express as px
import plotly.io as pio
pio.templates.default="plotly_white"
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


#readin the data

data=pd.read_csv('"C:/Users/levis/Downloads/Netflix-Subscriptions.csv"')
print(data.head())

data['Time Period']=pd.to_datetime(data['Time Period'], format='%d/%m/%Y')


print(data.head())

fig=go.Figure()

fig.add_trace(go.Scatter(x=data['Time Period'], 
                         y=data['Subscribers']
                         mode='lines',
                         name='Subscribers'))

fig.update_layout(title='Netflix Quarterly Subscriptions Growth',
                  xaxis_title='Date',
                  yaxis_title='Netflix Subscriptions')

fig.show()

#lets look at the yearly growth rate

data['Year']=data['Time Period'].dt.year
yearly_growth=data.groupby('Year')['Subscribers'].pct_change().fillna(0)*100

#create a new column for bar color 
data['Bar Color']=yearly_growth.apply(lambda x: 'green' if x>0 else 'red')

#plot the yearly subscriber growth rate using bar graphs

fig=go.Figure()
fig.add_trace(go.Bar(
    x=data['Year'],
    y=yearly_growth,
    marker_color=data['Bar Color'],
    name='Yearly Growth Rate'
))

fig.update_layout(title='Netflix Yearly Subscriber Growth Rate',
                  xaxis_title='Year',
                  yaxis_title='Yearly Growth Rate (%)')

fig.show()

#using ARIMA for forecasting Netflix quarterly subscriptions
#first convert the original dataframe into a time series format
#time period becomes the index and the subscribers becomes the data
time_series=data.set_index('Time Period')['Subscribers']

differenced_series=time_series.diff().dropna()

fig, axes=plt.subplots(1,2, figsize=(12,4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()

p,d,q=1,1,1

model=ARIMA(time_series, order=(p,d,q))

results=model.fit()

print(results.summary())


future_steps=5

predictions=results.predict(len(time_series), len(time_series)+future_steps-1)

predictions.astype(int)

#create a dataframe with original data and predictions

forecast=pd.DataFrame({'Original':time_series, 'Predictions': predictions})

#plot the original data and predictions

fig=go.Figure()

fig.add_trace(go.Scatter(x=forecast.index, 
                         y=forecast['Predictions'],
                         mode='lines',
                         name='Predictions'))
fig.add_trace(go.Scatter(x=forecast.index,
                         y=forecast['Original'],
                         mode='lines',
                         name='Original Data'))

fig.update_layout(title='Netflix Quarterly Subscription Predictions',
                  xaxis_title='Time Period',
                  yaxis_title='Subscriptions',
                  legend=dict(x=0.1, y=0.9),
                  showlegend=True)

fig.show()

