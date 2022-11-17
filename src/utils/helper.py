import yfinance as yf


def parse_command(user_msg_list: str) -> dict:
    user_msg_list = user_msg_list.split(" ")
    user_command_key = ["symbol", "command", "start_date", "end_date"]
    # Dict comprehension
    user_command = {user_command_key[i]: user_msg_list[i]
                    for i in range(len(user_msg_list))}

    return user_command


def data_loader(symbol, start_date):
    df = yf.Ticker(symbol).history(start=start_date)
    df['Date'] = df.index

    return df
