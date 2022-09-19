from flask import (Flask, request, abort, jsonify)
from flask_restful import Resource, Api
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage)
from dotenv import (load_dotenv, dotenv_values)


app = Flask(__name__)
load_dotenv()

# API
api = Api(app)
config = dotenv_values('.env')
Stock_list = []


class stock(Resource):
    def get(self, symbol):
        for stock in Stock_list:
            if stock['symbol'] == symbol:
                return {''}

    def post(self):
        data = request.get_json()
        return jsonify({data: data})


# LINE BOT
line_bot_api = LineBotApi(config['LINE_BOT_ACCESS_TOKEN'])
handler = WebhookHandler(config['LINE_BOT_SECRET'])


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
    print(event)
    if '!ma' in event.message.text:
        stonkCode = event.message.text.split(' ').pop()
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=stonkCode))
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='請輸入正確指令'))


if __name__ == "__main__":
    app.run()
