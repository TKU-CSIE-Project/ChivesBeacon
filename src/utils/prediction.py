from utils.helper import data_loader
from utils.featuring import featuring_kd
import matplotlib
from models.model import LSTMNET
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('src/')
import os
import pyimgur
# matplotlib.use("TkAgg")

CLIENT_ID = os.getenv('PYIMGUR_CLIENT_ID')

def normalize(df):
    data = df.copy()
    min_max_scaler = MinMaxScaler()

    data['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
    data['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
    data['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
    data['Volume'] = min_max_scaler.fit_transform(
        df.Volume.values.reshape(-1, 1))
    data['Close'] = min_max_scaler.fit_transform(
        df.Close.values.reshape(-1, 1))
    data['K'] = min_max_scaler.fit_transform(df.K.values.reshape(-1, 1))
    data['D'] = min_max_scaler.fit_transform(df.D.values.reshape(-1, 1))

    data = data.drop(['Symbol'], axis=1)
    data = data.drop(['Date'], axis=1)
    data = data.drop(['RSV'], axis=1)
    data = data.drop(['Dividends'], axis=1)
    data = data.drop(['Stock Splits'], axis=1)

    return data


def data_split(data, sample):
    # 將dataframe 轉成 numpy array
    data = data.values
    newdata = []
    y_data = []
    # 若想要觀察的 time_frame 為20天, 需要多加一天做為驗證答案
    for i in range(len(data)-sample):  # 從 datavalue 的第0個跑到倒數第 time_frame+1 個
        # 逐筆取出 time_frame+1 個 K棒數值做為一筆 instance
        newdata.append(data[i:(i+sample)])
        y_data.append(data[i+sample, -1:])

    n_train = round(0.8*len(newdata))  # 取 result 的前 80% instance做為訓練資料
    newdata = np.array(newdata)

    x_train = newdata[:int(n_train)]
    # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案
    y_train = y_data[:int(n_train)]
    x_test = newdata[int(n_train):]
    y_test = y_data[int(n_train):]

    # 將資料組成變好看一點
    x_train = x_train.reshape(-1, 30, 7)
    x_test = x_test.reshape(-1, 30, 7)
    return x_train, np.array(y_train), x_test, np.array(y_test)


def denormalize(df, norm_value):
    original_value = df['Close'].values.reshape(-1, 1)
    norm_value = norm_value.reshape(-1, 1)

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)

    return denorm_value


def prediction(symbol: str, start_date: str, end_date=None):
    data = data_loader(symbol, start_date, end_date)
    data = featuring_kd(data)
    historydf_norm = normalize(data)
    historydf_norm = historydf_norm.astype('float32')

    # 以30天為一區間進行股價預測
    x_train, y_train, x_test, y_test = data_split(historydf_norm, 30)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    model = LSTMNET(7, 64, 1)
    model.load_state_dict(torch.load('src/models/lstm_params.pkl'))
    model.eval()

    var_data = Variable(x_test)
    pred_test = model(var_data)
    pred_test = pred_test.view(-1).data.numpy()
    y_test = y_test.view(-1).data.numpy()

    denorm_pred = denormalize(data, pred_test)
    denorm_ytest = denormalize(data, y_test)
    plt.plot(denorm_pred, 'r', label='prediction')
    plt.plot(denorm_ytest, 'b', label='real')
    plt.legend(loc='best')

    im = pyimgur.Imgur(CLIENT_ID)
    picture = "src/cache/pre.png"
    plt.savefig(picture)
    plt.clf()
    uploaded_image = im.upload_image(picture, title="")
    return uploaded_image.link


