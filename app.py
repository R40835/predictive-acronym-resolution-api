from flask import Flask, request, render_template
from flask.json import jsonify
from gensim.models import KeyedVectors
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import datetime
import numpy as np

app = Flask(__name__)
fasttext_model = KeyedVectors.load('fasttext-wiki-news-subwords-300')
rnn_model = load_model('rnn_ner3.h5')
max_seq_length = 186
vocab_size = 999999
log_file_name = 'app.log'


@app.route('/predict', methods=['POST'])
def api_prediction():
    data = request.json
    try:
        sentence = data['sentence']
    except KeyError:
        return jsonify({'error': 'key "sentence" missing.'})
    words = sentence.split(' ')
    word_embeddings = [[fasttext_model[word] if word in fasttext_model else np.zeros(300) for word in words]]
    X = pad_sequences(word_embeddings, maxlen=max_seq_length, padding='post', value=vocab_size-1, dtype='float32')
    y = rnn_model.predict(X)
    y = np.argmax(y, axis=-1)
    y = y[y!=5]
    y = y.tolist()
    classes = {1: "B-AC", 3: "B-LF", 4: "I-LF", 0: "B-O"}
    y_pred = [classes[i] for i in y]

    # log inside try catch so that it doesn't interfere with the prediction function
    try:
        log_message(sentence, y_pred)
    except Exception as e:
        print(f"Internal server error: {e}")

    return jsonify({"result": y_pred})


@app.route('/', methods=['GET'])
def index():
    return "<h1>RNN for NER.</h1>"


@app.route('/logs')
def get_logs():
    logs = []
    try:
        with open(log_file_name, 'r') as file:
            content = file.read()
            if len(content)!=0: # empty file
                lines = content.split('\n')
                for line in lines:
                    if len(line) != 0: # empty line/str
                        log_elements = line.split('<*****>')
                        logs.append({
                            "datetime" : log_elements[0],
                            "sentence" : log_elements[1],
                            "predictions" : log_elements[2],
                        })
            return jsonify(logs = logs)
    except FileNotFoundError:
        return "Log file does not exist.", 404
    except Exception as e:
        return f"Internal server error: {e}", 500


def log_message(sentence, predictions):
    with open(log_file_name, 'a') as file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f"{timestamp}<*****>{sentence}<*****>{predictions}\n")


if __name__ == '__main__':
    app.run(debug=True)