from flask import (Flask, request, abort)
from flask_restful import Api
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage)
from dotenv import (load_dotenv)
from utils.helper import (
    parse_command, parse_recommend_command, compare_date, validate_date)
from controllers.indicatorControllers import IndicatorController
from controllers.predictionControllers import PredictionController
from controllers.recommendControllers import RecommendController
from configs.config import COMMAND_LIST
from resources.Stock import Stock
from resources.Recommend import Recommend
import datetime
import os


app = Flask(__name__)
load_dotenv()

# API
api = Api(app)
api.add_resource(Stock, '/v1/stock')
api.add_resource(Recommend, '/v1/recommend')


# LINE BOT
line_bot_api = LineBotApi(os.getenv('LINE_BOT_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_BOT_SECRET'))


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    input_array = event.message.text.split(" ")
    print(input_array)
    if len(input_array) < 3:
        user_input = parse_recommend_command(event.message.text)
        command = user_input.get('command', '').lower()
        start_date = user_input.get('start_date')

        # User input validation
        if validate_date(start_date) != True:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='日期格式錯誤'))
        elif compare_date(start_date, str(datetime.date.today())):
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='起始日不得大於今天日期'))

        # For command
        if command == 'recommend':
            recommend_list = RecommendController(start_date).stock_recommend()
            recommend_text = ''

            for i in range(len(recommend_list)):
                if (i == 9):
                    recommend_text += f"第{i+1}名: {recommend_list[i][0]}"
                else:
                    recommend_text += f"第{i+1}名: {recommend_list[i][0]}\n"

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=recommend_text))

    else:
        user_input = parse_command(event.message.text)
        symbol = user_input.get('symbol')
        command = user_input.get('command', '').lower()
        start_date = user_input.get('start_date')
        end_date = user_input.get(
            'end_date', datetime.date.today().strftime('%Y-%m-%d'))

        # User input validation
        if symbol == None:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='請輸入股票代號'))

        elif command not in COMMAND_LIST:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='請輸入正確指令'))

        elif start_date == None:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='請輸入起始日期'))

        elif validate_date(start_date) != True or validate_date(end_date) != True:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='日期格式錯誤'))

        elif compare_date(start_date, str(datetime.date.today())):
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='起始日不得大於今天日期'))

        # Command conditions
        elif command == 'kd':
            kd_link = IndicatorController(
                symbol, start_date, end_date).kd_graph()

            if kd_link == None:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text='查無該股票資料，請重新輸入股票代碼'))

            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    ImageSendMessage(original_content_url=kd_link, preview_image_url=kd_link))

        elif command == 'macd':
            macd_link = IndicatorController(
                symbol, start_date, end_date).macd_graph()
            print(macd_link)
            if macd_link == None:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text='查無該股票資料，請重新輸入股票代碼'))

            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    ImageSendMessage(original_content_url=macd_link, preview_image_url=macd_link))

        elif command == 'bias':
            bias_link = IndicatorController(
                symbol, start_date, end_date).bias_graph()

            if bias_link == None:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text='查無該股票資料，請重新輸入股票代碼'))

            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    ImageSendMessage(original_content_url=bias_link, preview_image_url=bias_link))

        elif command == 'bollinger':
            bollinger_link = IndicatorController(
                symbol, start_date, end_date).bollinger_band_graph()

            if bollinger_link == None:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text='查無該股票資料，請重新輸入股票代碼'))

            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    ImageSendMessage(original_content_url=bollinger_link, preview_image_url=bollinger_link))

        elif command == 'candle':
            candle_link = IndicatorController(
                symbol, start_date, end_date).candlestick_chart_graph()

            if candle_link == None:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text='查無該股票資料，請重新輸入股票代碼'))

            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    ImageSendMessage(original_content_url=candle_link, preview_image_url=candle_link))

        elif command == 'prediction':
            prediction_link = PredictionController(
                symbol, start_date, end_date).stock_predictions()

            if prediction_link == None:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text='查無該股票資料，請重新輸入股票代碼'))

            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    ImageSendMessage(original_content_url=prediction_link, preview_image_url=prediction_link))

        # Unexpected error
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='哎呀!可能有東西壞掉了QQ'))


if __name__ == "__main__":
    app.run(host="0.0.0.0")
