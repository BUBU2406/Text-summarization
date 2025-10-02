from flask import Flask, render_template, json, request, Response
import os
import update
import socket

# Connect to Redis

app = Flask(__name__)


# mod = Train()
@app.route("/")
def hello():
    # mod = Train()
    # if mod.predict_sentence("shop phuc vu kem, san pham khong duoc nhu mong doi :( ",'./data/word2vec.model2','./data/LSTM.model73') == 1:
    #   print("tieu cuc")
    # else:
    #   print("tic cuc")
    return render_template("index.html")


# @app.route("/welcome")
# def welcome():

#     return  json.dumps({'chuan':'van'})


@app.route("/postmt", methods=["POST"])
def post():
    # print( request.data.decode('utf-8'))
    text = request.data.decode("utf-8")
    print(text)

    contents = update.summarizations(text)
    return json.dumps({"result": contents})


# @app.route("/getrantext")
# def getText():
#     text =
#     return  json.dumps({'text':text , 'label' : lb})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
