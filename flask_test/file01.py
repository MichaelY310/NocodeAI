from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/firstapi',methods=['get'])
def hello():
    # 获取参数
    param = request.args.get('param')
    return f"hello:\t{param}"


def new_playground():
    # 获取参数
    param = request.args.get('param')
    return f"hello:\t{param}"


# host:指定绑定IP，port：指定绑定端口，debug指定：是否接受调试，是否返回错误信息
app.run(host='127.0.0.1', port=8080, debug=True)
