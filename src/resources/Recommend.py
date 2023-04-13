import sys
sys.path.append('src/')
import datetime
from controllers.recommendControllers import RecommendController
from flask import request
from flask_restful import Resource
from utils.helper import (validate_date, compare_date)



class Recommend(Resource):
    def post(self):
        data = request.get_json()
        date = data.get('date')

        # User input validation

        if date == None:
            return {'message': '請輸入日期', 'status_code': 4022}
        if validate_date(date) != True:
            return {'message': '日期格式錯誤', 'status_code': 4023}
        elif compare_date(date, str(datetime.date.today())):
            return {'message': '日期不得小於今天日期', 'status_code': 4024}

        # Command conditions

        else:
            recommend_list = RecommendController(date).stock_recommend()
            recommend_text = ''

            for i in range(len(recommend_list)):
                if (i == 9):
                    recommend_text += f"第{i+1}名: {recommend_list[i][0]}"
                else:
                    recommend_text += f"第{i+1}名: {recommend_list[i][0]}\n"
            if recommend_text == '':
                return {'message': 'Data Was To Old To Get Recommend List', 'status_code': 2023}

            return {'message': 'Success Get Recommend', 'recommend_list': recommend_text, 'status_code': 200}

        return {'message': 'Something Error', 'status_code': 5020}
