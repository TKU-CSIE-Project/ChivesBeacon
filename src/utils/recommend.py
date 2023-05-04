import numpy as np
import pandas as pd
import joblib
from scipy import stats
import sys
sys.path.append('src/')


def featuring_train(data):
    # string to datetime
    # data['Date'] = pd.to_datetime(data['Date'])
    data['Target'] = (data['Open']-data['Close'])/data['Open']
    data['Target'] = data['Target'].fillna(0)

    # Fill missing values
    cols = ['Open', 'High', 'Low', 'Close']
    data.loc[:, cols] = data.loc[:, cols].ffill()
    data.loc[:, cols] = data.loc[:, cols].bfill()

    # Calculate Daily_Range and Mean
    data['Daily_Range'] = data['Close'] - data['Open']
    data['Mean'] = (data['High']+data['Low']) / 2
    data['Mean'] = data['Mean'].astype(int)

    # Standardization
    data['Open'] = stats.zscore(data['Open'])
    data['High'] = stats.zscore(data['High'])
    data['Low'] = stats.zscore(data['Low'])
    data['Close'] = stats.zscore(data['Close'])
    data['Volume'] = stats.zscore(data['Volume'])
    data['Daily_Range'] = stats.zscore(data['Daily_Range'])
    data['Mean'] = stats.zscore(data['Mean'])
    data['Target'] = stats.zscore(data['Target'])

    # drop unused data
    data = data.drop(['Symbol'], axis=1)
    data = data.drop(['Dividends'], axis=1)
    data = data.drop(['Stock Splits'], axis=1)
    return data


def recommend(date: str):
    stock_prices = pd.read_csv('src/utils/company.csv')
    stock_prices = stock_prices.fillna(0)
    data = featuring_train(stock_prices)
    # split data
    data_test = data[data['Date'] > '2022-04-01']
    # reset index for test data
    data_test = data_test.reset_index(drop=True)
    # drop unused data
    data_test = data_test.drop(['Date'], axis=1)

    # Separation into learning features and objective variables
    X_test = data_test.drop(['Target'], axis=1)

    # load the model from disk
    loaded_model = joblib.load('src/models/LRmodel.learn')
    result = loaded_model.predict(X_test)
    stock_prices = stock_prices[stock_prices['Date'] > '2022-04-01']

    stock_prices['Predictions'] = result

    stock_prices = stock_prices[stock_prices['Date'] == date+' 00:00:00+08:00']
    print(stock_prices)
    stock_prices['Rank'] = stock_prices['Predictions'].rank(method='max')
    stock_prices = stock_prices.sort_values(by='Rank').head(10)

    return stock_prices.loc[:, ['Symbol', 'Rank']]
