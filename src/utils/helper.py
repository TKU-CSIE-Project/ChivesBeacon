import yfinance as yf
import requests
import pandas as pd

def parse_command(user_msg_list: str) -> dict:
    user_msg_list = user_msg_list.split(" ")
    user_command_key = ["symbol", "command", "start_date", "end_date"]
    # Dict comprehension
    user_command = {user_command_key[i]: user_msg_list[i]
                    for i in range(len(user_msg_list))}

    return user_command


def data_loader(symbol, start_date, end_date=None):
    df = yf.Ticker(symbol).history(start=start_date, end=end_date)
    df['Date'] = df.index

    return df

def get_symbol()->list:
    link = 'https://quality.data.gov.tw/dq_download_json.php?nid=11549&md5_url=bb878d47ffbe7b83bfc1b41d0b24946e'
    r = requests.get(link)
    df = pd.DataFrame(r.json())
    df = df.iloc[148:,0:]["證券代號"].values
    
    return df