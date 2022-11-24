import sys
sys.path.append('src/')
from utils.indicator import Indicators
import pandas as pd
pd.options.mode.chained_assignment = None

def featuring_kd(data):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist[0:3])):
        new_df = data[data['Symbol'] == symbollist[i]]
        ind_df = Indicators(new_df)
        ind_df.kv()
        ind_df.dv()
        concat_df = pd.concat([concat_df,new_df])

    return concat_df

def featuring_macd(data):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist)):
        new_df = data[data['Symbol'] == symbollist[i]]
        ind_df = Indicators(new_df)
        ind_df.macd()
        new_df = new_df.drop(['12_EMA','26_EMA','MACD_histogram'], axis=1)
        concat_df = pd.concat([concat_df,new_df])

    return concat_df

def featuring_ma(data,day=20):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist)):
        new_df = data[data['Symbol'] == symbollist[i]]
        ind_df = Indicators(new_df)
        ind_df.ma(day)
        concat_df = pd.concat([concat_df,new_df])

    return concat_df

def featuring_bias(data,day=6):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist)):
        new_df = data[data['Symbol'] == symbollist[i]]
        ind_df = Indicators(new_df)
        ind_df.bias(day)
        concat_df = pd.concat([concat_df,new_df])

    return concat_df

def featuring_bollinger_band(data,day=20):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist)):
        new_df = data[data['Symbol'] == symbollist[i]]
        ind_df = Indicators(new_df)
        ind_df.bollinger_band(day)
        new_df = new_df.drop(['Bollinger_mid'], axis=1)
        concat_df = pd.concat([concat_df,new_df])

    return concat_df

def shift_data(data):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist)):
        new_df = data[data['Symbol'] == symbollist[i]]
        new_df = new_df.shift(-20)
        new_df = new_df.dropna()
        concat_df = pd.concat([concat_df,new_df])
        
    concat_df.to_csv("src/utils/feature_company1.csv", index=False, encoding="utf_8_sig")

    return concat_df