from indicator import Indicators
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
        new_df = new_df.shift(-8)
        new_df = new_df.dropna()
        concat_df = pd.concat([concat_df,new_df])

    return concat_df