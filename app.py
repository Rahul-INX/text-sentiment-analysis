from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import tensorflow as tf
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, session
from flask_session import Session
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Configure session storage
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

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

    # Get the indices of the top 3 probabilities in descending order
    top_indices = np.argsort(predicted_probs)[::-1][:3]

    # Map the indices to the corresponding emotions and probabilities
    emotions = ['anger', 'fear', 'sadness', 'surprise', 'joy', 'love']
    top_emotions = [emotions[idx] for idx in top_indices]
    top_probabilities = [float(predicted_probs[idx]) for idx in top_indices]  # Convert to float

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current timestamp

    return top_emotions, top_probabilities, timestamp  # Return emotions, probabilities, and timestamp

def save_history(history):
    with open('history.json', 'w') as f:
        json.dump(history, f)

def load_history():
    if not os.path.exists('history.json'):
        with open('history.json', 'w') as f:
            f.write("[]")  # Write an empty list to the file
        return []  # Return an empty list as the history

    try:
        with open('history.json', 'r') as f:
            history = json.load(f)
    except:
        history = []
    return history

def predict_inferred_emotions(emotions, probabilities, threshold):
    inferred_emotions = []

    # Define the rules for inferring emotions based on the given emotions
    rules = [
    (('joy', 'love'), 'bliss'),
    (('surprise', 'love'), 'affection'),
    (('surprise', 'joy'), 'elation'),
    (('sadness', 'joy'), 'nostalgia'),
    (('sadness', 'love'), 'melancholy'),
    (('fear', 'joy'), 'excitement'),
    (('fear', 'love'), 'longing'),
    (('anger', 'joy'), 'exasperation'),
    (('anger', 'love'), 'passion'),
    (('sadness', 'surprise', 'love'), 'bittersweet'),
    (('sadness', 'surprise'), 'disappointment'),
    (('anger', 'surprise'), 'outrage'),
    (('fear', 'surprise'), 'anxiety'),
    (('surprise',), 'surprise'),
    (('anger', 'sadness'), 'resentment'),
    (('anger', 'fear', 'sadness'), 'resigned'),
    (('anger', 'fear'), 'frustration'),
    (('sadness', 'fear'), 'despair'),
    (('fear',), 'fear'),
    (('sadness',), 'sadness'),
    (('anger',), 'anger'),
    (('joy',), 'joy'),
    (('love',), 'love'),
    (('joy', 'surprise', 'love'), 'delight'),
    (('anger', 'surprise', 'joy'), 'indignation'),
    (('fear', 'sadness', 'joy'), 'admiration'),
    (('fear', 'sadness', 'love'), 'sorrow'),
    (('anger', 'fear', 'joy'), 'outrage'),
    (('anger', 'fear', 'love'), 'rage'),
    (('surprise', 'sadness', 'fear'), 'awe'),
    (('surprise', 'sadness', 'love'), 'amazement'),
    (('surprise', 'sadness', 'joy'), 'amusement'),
    (('fear', 'surprise', 'love'), 'trepidation'),
    (('anger', 'sadness', 'love'), 'heartache'),
    (('anger', 'surprise', 'sadness'), 'fury'),
    (('anger', 'surprise', 'fear'), 'hostility'),
    (('sadness', 'fear', 'love'), 'grief'),
    (('sadness', 'surprise', 'joy'), 'regret'),
    (('sadness', 'surprise', 'fear'), 'pity'),
    (('sadness', 'joy', 'love'), 'yearning'),
    (('anger', 'joy', 'love'), 'zeal')
    ]


    filtered_emotions = []

    # Filter emotions based on probability threshold
    for i in range(len(emotions)):
        if probabilities[i] >= threshold:
            filtered_emotions.append(emotions[i])

    # Check if any of the defined rules match the filtered emotions
    for rule_emotions, inferred_emotion in rules:
        if all(emotion in filtered_emotions for emotion in rule_emotions):
            inferred_emotions.append(inferred_emotion)

    # Remove atomic emotions from inferred emotions if its size is greater than 3
    if len(inferred_emotions) > 3:
        inferred_emotions = [emotion for emotion in inferred_emotions if emotion not in ('anger', 'fear', 'sadness', 'surprise', 'joy', 'love')]

    print("THRESHOLD:", threshold)
    print("FILTERED EMOTION:", filtered_emotions)
    print("INFERRED EMOTION:", inferred_emotions)
    return inferred_emotions

def get_random_string():
    with open('test_text.txt', 'r') as file:
        lines = file.readlines()
        random_string = random.choice(lines).strip()
        return random_string


history = load_history()

@app.route('/', methods=['GET', 'POST'])
def index():
    inferred_emotions = []
    
    if 'threshold' in session:
        threshold = session['threshold']  # Get the threshold value from the session

    random_string = session.pop('random_string', '')  # Get and remove the random_string value from the session or assign an empty string

    if request.method == 'POST':
        text = request.form['text']
        emotions, probabilities, timestamp = predict_emotion(text)
        inferred_emotions = predict_inferred_emotions(emotions, probabilities, threshold)
        entry = {
            'text': text,
            'emotions': emotions,
            'probabilities': probabilities,
            'inferred_emotions': inferred_emotions,
            'timestamp': timestamp
        }
        history.insert(0, entry)
        save_history(history)
        return redirect('/')
    
    return render_template('index.html', history=history, inferred_emotions=inferred_emotions, threshold=threshold, random_string=random_string)



@app.route('/clear', methods=['POST'])
def clear_history():
    history.clear()
    save_history(history)
    return redirect('/')

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    threshold = float(request.form['threshold'])
    session['threshold'] = threshold  # Update the threshold value in the session
    return redirect('/')
@app.route('/random', methods=['POST'])
def random_text():
    random_string = get_random_string()
    session['random_string'] = random_string
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
