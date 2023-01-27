import sys
sys.path.append('src/')
from configs.config import COMMAND_LIST
from utils.helper import compare_date
from flask_restful import Resource
from flask import request
from controllers.indicatorControllers import IndicatorController
import datetime


class Stock(Resource):
    def post(self):
        data = request.get_json()
        symbol = data.get('symbol')
        command = data.get('command', '').lower()
        start_date = data.get('start_date')
        end_date = data.get(
            'end_date', datetime.date.today().strftime('%Y-%m-%d'))

        # User input validation
        if symbol == None:
            return {'message': '請輸入股票代號', 'status_code': 4020}

        elif command not in COMMAND_LIST:
            return {'message': '請輸入正確指令', 'status_code': 4021}

        elif start_date == None:
            return {'message': '請輸入起始日期', 'status_code': 4022}

        elif compare_date(start_date, str(datetime.date.today())):
            return {'message': '起始日不得小於今天日期', 'status_code': 4023}

        # Command conditions
        else:
            if command == 'kd':
                kd_link = IndicatorController(
                    symbol, start_date, end_date).kd_graph()
                if kd_link == None:
                    return {'message': 'Empty Stock Data', 'status_code': 4024}
                else:
                    return {'message': 'Success Get KD', 'img_link': kd_link, 'status_code': 200}

            elif command == 'macd':
                macd_link = IndicatorController(
                    symbol, start_date, end_date).macd_graph()
                if macd_link == None:
                    return {'message': 'Empty Stock Data', 'status_code': 4024}
                else:
                    return {'message': 'Success Get MACD', 'img_link': macd_link, 'status_code': 200}

            elif command == 'bias':
                bias_link = IndicatorController(
                    symbol, start_date, end_date).bias_graph()
                if bias_link == None:
                    return {'message': 'Empty Stock Data', 'status_code': 4024}
                else:
                    return {'message': 'Success Get BIAS', 'img_link': bias_link, 'status_code': 200}

            elif command == 'bollinger':
                bollinger_link = IndicatorController(
                    symbol, start_date, end_date).bollinger_band_graph()
                if bollinger_link == None:
                    return {'message': 'Empty Stock Data', 'status_code': 4024}
                else:
                    return {'message': 'Success Get Bollinger', 'img_link': bollinger_link, 'status_code': 200}

            elif command == 'candle':
                candle_link = IndicatorController(
                    symbol, start_date, end_date).candlestick_chart_graph()
                if candle_link == None:
                    return {'message': 'Empty Stock Data', 'status_code': 4024}
                else:
                    return {'message': 'Success Get Candle', 'img_link': candle_link, 'status_code': 200}

        return {'message': 'Something Error', 'status_code': 5020}
