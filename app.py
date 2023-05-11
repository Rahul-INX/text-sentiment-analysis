from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the emotion trained model
model = tf.keras.models.load_model('emotion_model_trained.h5')

def preprocess_text(text):
    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])

    # Pad the sequence
    padded_sequence = pad_sequences(sequence, truncating='post', maxlen=50, padding='post')

    return padded_sequence

def predict_emotion(text):
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Predict the emotion probabilities
    predicted_probs = model.predict(processed_text)[0]

    # Get the index of the predicted emotion
    predicted_index = np.argmax(predicted_probs)

    # Map the predicted index to the corresponding emotion
    emotions = ['anger', 'fear', 'sadness', 'surprise', 'joy', 'love']
    predicted_emotion = emotions[predicted_index]

    return predicted_emotion

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        emotion = predict_emotion(text)
        return render_template('index.html', emotion=emotion)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
