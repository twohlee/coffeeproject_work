from flask import Flask, render_template
import pickle
import json

app = Flask(__name__)

@app.route('/')
def home():
    with open('./static/result/prediction_for_web.pickle', 'rb') as f:
        prediction_for_web = json.loads(pickle.load(f))
    return render_template('index.html', img = 'static/result/img.png', img_labeled = 'static/result/img_labeled.png', prediction_for_web = prediction_for_web)



@app.route('/count')
def count():
    with open('./static/result/prediction_for_web.pickle', 'rb') as f:
        prediction_for_web = json.loads(pickle.load(f))
    return render_template('count.html', img = 'static/result/img.png', img_labeled = 'static/result/img_labeled.png', prediction_for_web = prediction_for_web)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)