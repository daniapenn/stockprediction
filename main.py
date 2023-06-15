import streamlit as st
from datetime import date 
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Choose the start date of the data you wanna grab
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

# Drop down box
stocks = ("AAPL", "GOOG", "MSFT", "ADSK")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Slider
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# ticker is stock code
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    # puts date in first column
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Loading Data...done!")

# Data Table
st.subheader('Raw data')
# only show the last couple of days
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    #stock_open line
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    #stock_close line
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Data Graph
plot_raw_data()

# Forecasting
# grab date and close data
df_train = data[['Date', 'Close']]

# need to rename the columns of data to prophet specifications
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m=Prophet()
# need to fit the data set in order for prophet to be able to forecast
m.fit(df_train)
# data-times to be predicted based off number of periods wanted
future = m.make_future_dataframe(periods=period)
# forecast/predict the price of the periods given 
forecast = m.predict(future)

# Forecast Data Table
st.subheader('Forecast data')
st.write(forecast.tail())

# Forecast Data Graph
st.subheader('Forecast prediction')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

fig2 = m.plot_components(forecast)
st.write(fig2)


