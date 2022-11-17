import sys
sys.path.append('src/')
from utils.indicator import Indicators
from utils.helper import data_loader



class IndicatorController:
    def __init__(self, symbol: str, start_date: str, end_date=None):
        self.indicators = Indicators(data_loader(
            symbol, start_date, end_date=end_date))
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def KD(self):
        self.indicators.kd_line(date=self.start_date)


