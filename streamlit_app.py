import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page config
st.set_page_config(page_title="Stock Momentum Analysis", layout="wide")

# Title
st.title("Stock Momentum Analysis App")
st.markdown("An application by Sampreeth Shetty to fit LSTM, ARCH, and GARCH models on stock data")

# Streamlit input fields
ticker = st.text_input("Enter the stock ticker:", "AAPL")
start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2023-01-01"))

# Function to plot results
def plot_results(original, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(original.index, original, label='Actual')
    plt.plot(predicted.index, predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close')
    plt.legend()
    st.pyplot(plt.gcf())

# Download data
if st.button("Download Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        st.error("No data found for the given ticker and date range.")
    else:
        st.write("Data downloaded successfully!")
        st.dataframe(data)

        # Impute missing values with mean
        data['Adj Close'].fillna(data['Adj Close'].mean(), inplace=True)

        # Preprocessing data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                X.append(a)
                Y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 60
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

        # Predictions using LSTM
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Create dataframes for plotting LSTM results
        train_predict_df = pd.DataFrame(train_predict, index=data.index[time_step:train_size+time_step], columns=['Adj Close'])
        test_predict_df = pd.DataFrame(test_predict, index=data.index[train_size+time_step+1:len(data)-1], columns=['Adj Close'])

        # Plot LSTM results
        st.subheader("LSTM Model")
        plot_results(data['Adj Close'][time_step:train_size+time_step], train_predict_df['Adj Close'], "Train Data - LSTM Model")
        plot_results(data['Adj Close'][train_size+time_step+1:len(data)-1], test_predict_df['Adj Close'], "Test Data - LSTM Model")

        # ARCH model
        st.subheader("ARCH Model")
        arch_mod = arch_model(data['Adj Close'], vol='ARCH', p=1)
        arch_res = arch_mod.fit(disp="off")
        st.text(arch_res.summary())

        arch_forecast = arch_res.forecast(horizon=len(test_data))
        arch_pred = pd.DataFrame(np.sqrt(arch_forecast.variance.values[-1, :]), index=data.index[train_size+time_step+1:len(data)-1], columns=['Adj Close'])
        
        plot_results(data['Adj Close'][train_size+time_step+1:len(data)-1], arch_pred['Adj Close'], "ARCH Model Forecast")

        # GARCH model
        st.subheader("GARCH Model")
        garch_mod = arch_model(data['Adj Close'], vol='Garch', p=1, q=1)
        garch_res = garch_mod.fit(disp="off")
        st.text(garch_res.summary())

        garch_forecast = garch_res.forecast(horizon=len(test_data))
        garch_pred = pd.DataFrame(np.sqrt(garch_forecast.variance.values[-1, :]), index=data.index[train_size+time_step+1:len(data)-1], columns=['Adj Close'])
        
        plot_results(data['Adj Close'][train_size+time_step+1:len(data)-1], garch_pred['Adj Close'], "GARCH Model Forecast")
