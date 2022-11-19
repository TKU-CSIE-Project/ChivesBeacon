import sys
sys.path.append('src/')
from utils.helper import data_loader
from utils.indicator import Indicators


class IndicatorController:
    def __init__(self, symbol: str, start_date: str, end_date: str = None):
        self.indicators = Indicators(data_loader(
            symbol, start_date, end_date=end_date))
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def kd_graph(self):
        self.indicators.kd_line(start_date=self.start_date)

    def macd_graph(self):
        self.indicators.macd_line(start_date=self.start_date)

    def bias_graph(self):
        self.indicators.bias_line(start_date=self.start_date)

    def bollinger_band_graph(self):
        self.indicators.bollinger_band_line(start_date=self.start_date)

    def candlestick_chart_graph(self):
        self.indicators.candlestick_chart(start_date=self.start_date)
