from indicator import Indicators
import pandas as pd
pd.options.mode.chained_assignment = None

def featuring_kd(data):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist)):
        new_df = data[data['Symbol'] == symbollist[i]]
        ind_df = Indicators(new_df)
        ind_df.kv()
        ind_df.dv()
        new_df = new_df.shift(-8)
        new_df = new_df.dropna()
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
        new_df = new_df.shift(-8)
        new_df = new_df.dropna()
        concat_df = pd.concat([concat_df,new_df])

    return concat_df

def featuring_ma(data,day=20):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist)):
        new_df = data[data['Symbol'] == symbollist[i]]
        ind_df = Indicators(new_df)
        ind_df.ma(day)
        new_df = new_df.shift(-8)
        new_df = new_df.dropna()
        concat_df = pd.concat([concat_df,new_df])

    return concat_df

def featuring_bias(data,day=6):
    symbollist = sorted(list(set(data.Symbol.values)))
    concat_df = pd.DataFrame()

    for i in range(len(symbollist)):
        new_df = data[data['Symbol'] == symbollist[i]]
        ind_df = Indicators(new_df)
        ind_df.bias(day)
        new_df = new_df.shift(-8)
        new_df = new_df.dropna()
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
        new_df = new_df.shift(-8)
        new_df = new_df.dropna()
        concat_df = pd.concat([concat_df,new_df])

    return concat_df
