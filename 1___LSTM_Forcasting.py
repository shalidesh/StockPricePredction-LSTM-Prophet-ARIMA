
#import libruaries
import streamlit as st
from datetime import date
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.models import load_model
import math
from sklearn.metrics import mean_squared_error
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="💰",
    
)

st.sidebar.success("Select a page above.")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecasting')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

days = ('10', '20', '30', '40')
selected_day = st.selectbox('Select Days for prediction', days)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

model = load_model(f"models/LSTM_{selected_stock}.h5")
count=len(data)

data1=data.loc[:,["Date","Close"]]
data1['Date'] =  pd.to_datetime(data1['Date'])
data1['Date'] = data1['Date'].dt.tz_localize(None)
data1=data1.set_index('Date')


scaler=MinMaxScaler(feature_range=(0,1))
data1=scaler.fit_transform(np.array(data1).reshape(-1,1))

x_input=data1[-100:,:].reshape(1,-1)
print(x_input.shape)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

data_load_state1 = st.text('Predicting data...')
#demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
count=int(selected_day)
while(i<count):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
data_load_state1.text('Predicting data... done!')


print(lst_output)

day_new=np.arange(1,101)
day_pred=np.arange(101,100+count+1)

plt.plot(day_new,scaler.inverse_transform(data1[1947:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.subheader(f'PREDICTION FOR NEXT {count} DAYS')
st.pyplot()

df3=data1.tolist()
df3.extend(lst_output)
plt.plot(df3[1600:])
st.pyplot()

df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
st.pyplot()
