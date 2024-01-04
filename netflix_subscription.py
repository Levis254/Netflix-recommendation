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