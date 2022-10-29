from cmath import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import pyimgur
import yfinance as yf

CLIENT_ID = "Your_applications_client_id"
PATH = "A Filepath to an image on your computer"


class Indicators:
    '''
    input a company's dataframe to create indicators series
    :return series
    '''

    def __init__(self, data):
        self.__data = data

    def rsv(self):
        '''
        rsv = (今日收盤價 - 最近九天的最低價)/(最近九天的最高價 - 最近九天最低價)
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
        當日K值=前一日K值 * 2/3 + 當日RSV * 1/3
        '''
        data = self.__data
        if 'RSV' not in data:
            self.rsv()
        rsv = self.__data['RSV'].tolist()

        kv = [20 for _ in range(8)]
        ktemp = kv[0]
        for i in range(len(rsv) - 8):
            ktemp = ktemp * (2 / 3) + rsv[i + 8] * (1 / 3)
            kv.append(round(ktemp, 2))
        self.__data['K'] = kv

    def dv(self):
        '''
        當日D值=前一日D值 * 2/3 + 當日K值 * 1/3
        '''
        data = self.__data
        if 'K' not in data:
            self.kv()
        kv = self.__data['K'].tolist()

        dv = [50 for _ in range(8)]
        dtemp = dv[0]
        for i in range(len(kv) - 8):
            dtemp = dtemp * (2 / 3) + kv[i + 8] * (1 / 3)
            dv.append(round(dtemp, 2))
        self.__data['D'] = dv

    def macd(self):
        '''
        EMA(指數移動平均)：計算MACD時，會先計算長、短天期的指數移動平均線(EMA)
        ，一般來說短期常使用12日(n=12)、長期為26日(m=26)
        DIF = 12日EMA – 26日EMA
        MACD =快線取9日EMA
        柱狀圖(直方圖) = 快線–慢線，EMA快慢線相減後，得出來的差額就是在MACD圖形看到的柱狀圖。
        :return:
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

    def kd_line(self, date: str, savefig: bool = True):
        '''
        Make KD indicator's picture
        '''
        data = self.__data

        if 'K' not in data:
            self.kv()
        if 'D' not in data:
            self.dv()

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > date]
        data['K'].plot()
        data['D'].plot()
        plt.legend()
        plt.title('KD')

        im = pyimgur.Imgur('7055605c8712cfc')

        if savefig == True:
            picture = "src/cache/KD.png"
            plt.savefig(picture)
            uploaded_image = im.upload_image(
                picture, title="Uploaded with PyImgur")

            return uploaded_image.link

        else:
            plt.show()

    def macd_line(self, date: str, savefig: bool = True):
        '''
        Make MACD indicator's picture
        '''
        data = self.__data

        if 'DIF' not in data or 'MACD' not in data:
            self.macd()

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > date]

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

        if savefig == True:
            picture = "src/cache/MACD.png"
            plt.savefig(picture)
        else:
            plt.show()

    def bias_line(self, date: str, savefig: bool = True):
        data = self.__data

        if 'Bias' not in data:
            self.bias()

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > date]
        data['Bias'].plot(color='red')
        plt.legend()
        plt.title('Bias')

        if savefig == True:
            picture = "src/cache/Bias.png"
            plt.savefig(picture)
        else:
            plt.show()

    def bollinger_band_line(self, date: str, savefig: bool = True):
        data = self.__data

        if 'Bollinger_top' not in data or 'Bollinger_mid' not in data or 'Bollinger_down' not in data:
            self.bollinger_band()

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > date]

        data['Bollinger_top'].plot(color='red')
        data['Bollinger_mid'].plot(color='blue')
        data['Bollinger_down'].plot(color='green')
        data['Close'].plot(color='orange')
        plt.legend()
        plt.title('Bollinger_band')

        if savefig == True:
            picture = "src/cache/Bollinger_band.png"
            plt.savefig(picture)
        else:
            plt.show()

    def candlestick_chart(self, date: str, savefig: bool = True):
        data = self.__data

        data.index = pd.DatetimeIndex(data['Date'])
        data = data[data.index > date]
        mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

        if savefig == True:
            # 5b,20o,60g,120r,240p
            mpf.plot(data, style=s, type='candle', volume=True, mav=(5, 20, 60, 120, 240),
                     savefig='picture/' + str(round(data['證券代號'].values[0])) + '.png')
        else:
            mpf.plot(data, style=s, type='candle',
                     volume=True, mav=(5, 20, 60, 120, 240))

    def __str__(self):
        return self.__data.__str__()


