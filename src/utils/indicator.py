from cmath import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import pyimgur


CLIENT_ID = os.getenv('PYIMGUR_CLIENT_ID')
plt.switch_backend('agg')


class Indicators:
    '''
    input a company's dataframe to create indicators series
    :return series
    '''

    def __init__(self, data):
        self.__data = data

    def rsv(self):
        '''
        rsv = [(today's Close) - (the last nine days's Low)] / [(the last nine days's High) - (the last nine days's Low)]
        '''
        data = self.__data
        rsv = (
            data['Close'] - data['Close'].rolling(window=9).min()
        ) / (
            data['Close'].rolling(window=9).max() -
            data['Close'].rolling(window=9).min()
        ) * 100
        rsv = np.nan_to_num(rsv)
        self.__data['RSV'] = rsv

    def kv(self):
        '''
        Today's K = (K the day before) * 2/3 + (today's RSV) * 1/3
        '''
        data = self.__data
        if 'RSV' not in data:
            self.rsv()
        rsv = np.array(self.__data['RSV'].tolist())

        kv = np.array([50 for _ in range(8)])
        ktemp = kv[0]
        for i in range(len(rsv) - 8):
            ktemp = ktemp * (2 / 3) + rsv[i + 8] * (1 / 3)
            kv.append(round(ktemp, 2))
        self.__data['K'] = kv

    def dv(self):
        '''
        Today's D = (D the day before) * 2/3 + (today's K) * 1/3
        '''
        data = self.__data
        if 'K' not in data:
            self.kv()
        kv = np.array(self.__data['K'].tolist())

        dv = np.array([50 for _ in range(8)])
        dtemp = dv[0]
        for i in range(len(kv) - 8):
            dtemp = dtemp * (2 / 3) + kv[i + 8] * (1 / 3)
            dv.append(round(dtemp, 2))
        self.__data['D'] = dv

    def macd(self):
        '''
        EMA(n) = (EMA the day before(n) * (n-1) + today's Close * 2) ÷ (n+1)
        EMA(m) = (EMA the day before(m) * (m-1) + today's Close * 2) ÷ (m+1)
        DIF = EMA(n) - EMA(m)
        MACD(x) = (MACD the day before * (x-1) + DIF * 2) ÷ (x+1)
        '''
        data = self.__data
        data['12_EMA'] = data['Close'].ewm(span=12).mean()
        data['26_EMA'] = data['Close'].ewm(span=26).mean()
        data['DIF'] = data['12_EMA'] - data['26_EMA']
        data['MACD'] = data['DIF'].ewm(span=9).mean()
        data['MACD_histogram'] = data['DIF'] - data['MACD']

    def ma(self, day=20):
        data = self.__data
        data['{}_MA'.format(day)] = data['Close'].rolling(day).mean()
        data['{}_MA'.format(day)] = np.nan_to_num(data['{}_MA'.format(day)])

    def bias(self, day=6):
        data = self.__data
        if '{}_MA'.format(day) not in data:
            self.ma(day)

        data['Bias'] = 100 * (data['Close'] - data['{}_MA'.format(day)]
                              ) / data['Close'].rolling(day).mean()

    def bollinger_band(self, day=20):
        data = self.__data
        if '{}_MA'.format(day) not in data:
            self.ma(day)
        std = data['Close'].rolling(day).std()

        data['Bollinger_top'] = data['{}_MA'.format(day)] + std * 2
        data['Bollinger_mid'] = data['{}_MA'.format(day)]
        data['Bollinger_down'] = data['{}_MA'.format(day)] - std * 2

    def kd_line(self, start_date: str):
        '''
        Make KD indicator's picture
        '''
        data = self.__data

        if 'K' not in data:
            self.kv()
        if 'D' not in data:
            self.dv()

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > start_date]
        data['K'].plot()
        data['D'].plot()
        plt.legend()
        plt.title('KD')

        im = pyimgur.Imgur(CLIENT_ID)

        picture = "src/cache/KD.png"
        plt.savefig(picture)
        plt.clf()
        uploaded_image = im.upload_image(picture, title="")
        return uploaded_image.link

    def macd_line(self, start_date: str):
        '''
        Make MACD indicator's picture
        '''
        data = self.__data

        if 'DIF' not in data or 'MACD' not in data:
            self.macd()

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > start_date]

        data['MACD'].plot(kind='line')
        data['DIF'].plot(kind='line')
        for index, row in data.iterrows():
            if (row['MACD_histogram'] > 0):
                plt.bar(row['Date'], row['MACD_histogram'],
                        width=0.5, color='red')
            else:
                plt.bar(row['Date'], row['MACD_histogram'],
                        width=0.5, color='green')

        plt.legend()
        plt.title('MACD')

        im = pyimgur.Imgur(CLIENT_ID)

        picture = "src/cache/MACD.png"
        plt.savefig(picture)
        plt.clf()
        uploaded_image = im.upload_image(
            picture, title="Uploaded with PyImgur")

        return uploaded_image.link

    def bias_line(self, start_date: str):
        data = self.__data

        if 'Bias' not in data:
            self.bias()

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > start_date]
        data['Bias'].plot(color='red')
        plt.legend()
        plt.title('Bias')

        im = pyimgur.Imgur(CLIENT_ID)

        picture = "src/cache/BIAS.png"
        plt.savefig(picture)
        plt.clf()
        uploaded_image = im.upload_image(
            picture, title="Uploaded with PyImgur")

        return uploaded_image.link

    def bollinger_band_line(self, start_date: str):
        data = self.__data

        if 'Bollinger_top' not in data or 'Bollinger_mid' not in data or 'Bollinger_down' not in data:
            self.bollinger_band()

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > start_date]

        data['Bollinger_top'].plot(color='red')
        data['Bollinger_mid'].plot(color='blue')
        data['Bollinger_down'].plot(color='green')
        data['Close'].plot(color='orange')
        plt.legend()
        plt.title('Bollinger_band')

        im = pyimgur.Imgur(CLIENT_ID)

        picture = "src/cache/Bollinger_Band.png"
        plt.savefig(picture)
        plt.clf()
        uploaded_image = im.upload_image(
            picture, title="Uploaded with PyImgur")

        return uploaded_image.link

    def candlestick_chart(self, start_date: str):
        data = self.__data

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > start_date]
        mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

        picture = "src/cache/candlestick_chart.png"
        # 5b,20o,60g,120r,240p
        mpf.plot(data, style=s, type='candle', volume=True, mav=(5, 20, 60, 120, 240),
                 savefig=picture)

        im = pyimgur.Imgur('7055605c8712cfc')

        uploaded_image = im.upload_image(
            picture, title="Uploaded with PyImgur")

        return uploaded_image.link

    def __str__(self):
        return self.__data.__str__()

