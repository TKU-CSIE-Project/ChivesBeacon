from flask import (Flask, request, abort)
from flask_restful import Api
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage)
from dotenv import (load_dotenv)
from utils.helper import (parse_command, compare_date)
from controllers.indicatorControllers import IndicatorController
from configs.config import COMMAND_LIST
import datetime
import os


app = Flask(__name__)
load_dotenv()

# API
api = Api(app)


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
    user_input = parse_command(event.message.text)
    symbol = user_input.get('symbol')
    command = user_input.get('command').lower()
    start_date = user_input.get('start_date')
    end_date = user_input.get(
        'end_date', datetime.date.today().strftime('%Y-%m-%d'))

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
    elif compare_date(start_date, str(datetime.date.today())):
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='起始日不得小於今天日期'))
    # TODO: create command condition after here
    else:
        kd_link = IndicatorController(symbol, start_date, end_date).kd_graph()
        line_bot_api.reply_message(
            event.reply_token,
            ImageSendMessage(original_content_url=kd_link, preview_image_url=kd_link))


if __name__ == "__main__":
    app.run(host="0.0.0.0")
