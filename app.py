from flask import Flask, render_template, request
from bert_model import predict_sentiment  # 👈 Import the new BERT model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_sentiment(text)
    return render_template('index.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
