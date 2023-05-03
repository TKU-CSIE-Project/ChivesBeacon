import sys
sys.path.append('src/')
from utils.prediction import prediction


class PredictionController:
    def __init__(self, symbol: str, start_date: str, end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def stock_predictions(self):
        return prediction(self.symbol, self.start_date, self.end_date)
