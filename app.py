from flask import Flask, request, render_template
from flask.json import jsonify
from gensim.models import KeyedVectors
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import time
import numpy as np

app = Flask(__name__)
fasttext_model = KeyedVectors.load('fasttext-wiki-news-subwords-300')
rnn_model = load_model('rnn_ner3.h5')
max_seq_length = 186
vocab_size = 999999


@app.route('/predict', methods=['POST'])
def api_prediction():
    data = request.json
    try:
        sentence = data['sentence']
    except KeyError:
        return jsonify({'error': 'key "sentence" missing.'})
    sentence = sentence.split(' ')
    word_embeddings = [[fasttext_model[word] if word in fasttext_model else np.zeros(300) for word in sentence]]
    X = pad_sequences(word_embeddings, maxlen=max_seq_length, padding='post', value=vocab_size-1, dtype='float32')
    y = rnn_model.predict(X)
    y = np.argmax(y, axis=-1)
    y = y[y!=5]
    y = y.tolist()
    classes = {1: "B-AC", 3: "B-LF", 4: "I-LF", 0: "B-O"}
    y_pred = [classes[i] for i in y]
    return jsonify({"result": y_pred})


@app.route('/', methods=['GET'])
def index():
    return "<h1>RNN for NER.</h1>"


if __name__ == '__main__':
    app.run(debug=True)
