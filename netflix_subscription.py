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