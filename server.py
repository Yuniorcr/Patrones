import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flask import Flask, request, jsonify, json
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    palabra = data.get('sentence')
    sentence = [palabra]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=100, padding='post' ,truncating='post')
    pred = model.predict(padded)
    return json.dumps({"status": True, "msg": "OK", "data": pred.tolist()[0][0]}, ensure_ascii=False)
@app.route('/predict2/<sentence2>', methods=['GET'])
def predict2(sentence2):
    sentence = [sentence2]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=100, padding='post' ,truncating='post')
    pred = model.predict(padded)
    return json.dumps({"status": True, "msg": "OK", "data": pred.tolist()[0][0]}, ensure_ascii=False)

if __name__ == '__main__':
    # load model 
    model = tf.keras.models.load_model('Cybebullyng.h5')
    import pickle

    # loading tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    app.run()