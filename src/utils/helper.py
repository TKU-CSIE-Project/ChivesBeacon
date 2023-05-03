import yfinance as yf
import pandas as pd
import requests
import time
from datetime import date
from dateutil.parser import parse


def parse_recommend_command(user_msg_list: str) -> dict:
    user_msg_list = user_msg_list.split(" ")
    user_command_key = ["command", "start_date"]
    # Dict comprehension
    user_command = {user_command_key[i]: user_msg_list[i]
                    for i in range(len(user_msg_list))}

    return user_command


def parse_command(user_msg_list: str) -> dict:
    user_msg_list = user_msg_list.split(" ")
    user_command_key = ["symbol", "command", "start_date", "end_date"]
    # Dict comprehension
    user_command = {user_command_key[i]: user_msg_list[i]
                    for i in range(len(user_msg_list))}

    return user_command


def data_loader(symbol: str, start_date: str, end_date=None):
    df = yf.Ticker(symbol).history(start=start_date, end=end_date)
    # df = df.drop(['Dividends', 'Stock Splits'], axis=1)
    df['Symbol'] = symbol[:-3]
    df['Date'] = df.index
    if df.empty == True:
        return df
    elif df.empty == False:
        # df = df.drop(['Dividends', 'Stock Splits'], axis=1)
        df['Symbol'] = symbol[:-3]
        df['Date'] = df.index

        return df


def compare_date(first_date, second_date):
    formatted_date1 = time.strptime(first_date, "%Y-%m-%d")
    formatted_date2 = time.strptime(second_date, "%Y-%m-%d")

    return (formatted_date1 > formatted_date2)


def get_delta_day(date1, date2):
    date1_list = date1.split('-')
    date2_list = date2.split('-')
    date1_date = date(int(date1_list[0]), int(
        date1_list[1]), int(date1_list[2]))
    date2_date = date(int(date2_list[0]), int(
        date2_list[1]), int(date2_list[2]))
    delta = date2_date - date1_date

    return delta.days


def validate_date(date):
    try:
        parse(date, fuzzy=False)
        return True

    except ValueError:
        return False


def get_symbol() -> list:
    link = 'https://quality.data.gov.tw/dq_download_json.php?nid=11549&md5_url=bb878d47ffbe7b83bfc1b41d0b24946e'
    r = requests.get(link)
    df = pd.DataFrame(r.json())
    df = df[df['證券代號'] >= '1101']["證券代號"].values

    return df


def all_company_data(start_date: str, end_date: str = None):
    symbol_list = get_symbol()
    df = data_loader(get_symbol()[0]+".TW", start_date, end_date)
    for i in range(len(symbol_list)-1):
        try:
            df1 = data_loader(symbol_list[i+1]+".TW", start_date, end_date)
            df = pd.concat([df, df1])
        except:
            print("Error: empty dataframe", symbol_list[i+1])
    df.to_csv("src/utils/company.csv", index=False, encoding="utf_8_sig")

    return df
