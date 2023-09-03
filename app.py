from flask import Flask, render_template, request, redirect, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import pickle
import random
import json
from datetime import datetime

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model('best_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set the inference threshold (initial value)
threshold = 0.5

# Define complex emotions based on rules
complex_emotions = {
    ('joy', 'love'): 'bliss',
    ('surprise', 'love'): 'affection',
    ('surprise', 'joy'): 'elation',
    ('sadness', 'joy'): 'nostalgia',
    ('sadness', 'love'): 'melancholy',
    ('fear', 'joy'): 'excitement',
    ('fear', 'love'): 'longing',
    ('anger', 'joy'): 'exasperation',
    ('anger', 'love'): 'passion',
    ('sadness', 'surprise', 'love'): 'bittersweet',
    ('sadness', 'surprise'): 'disappointment',
    ('anger', 'surprise'): 'outrage',
    ('fear', 'surprise'): 'anxiety',
    ('surprise',): 'surprise',
    ('anger', 'sadness'): 'resentment',
    ('anger', 'fear', 'sadness'): 'resigned',
    ('anger', 'fear'): 'frustration',
    ('sadness', 'fear'): 'despair',
    ('fear',): 'fear',
    ('sadness',): 'sadness',
    ('anger',): 'anger',
    ('joy',): 'joy',
    ('love',): 'love',
    ('joy', 'surprise', 'love'): 'delight',
    ('anger', 'surprise', 'joy'): 'indignation',
    ('fear', 'sadness', 'joy'): 'admiration',
    ('fear', 'sadness', 'love'): 'sorrow',
    ('anger', 'fear', 'joy'): 'outrage',
    ('anger', 'fear', 'love'): 'rage',
    ('surprise', 'sadness', 'fear'): 'awe',
    ('surprise', 'sadness', 'love'): 'amazement',
    ('surprise', 'sadness', 'joy'): 'amusement',
    ('fear', 'surprise', 'love'): 'trepidation',
    ('anger', 'sadness', 'love'): 'heartache',
    ('anger', 'surprise', 'sadness'): 'fury',
    ('anger', 'surprise', 'fear'): 'hostility',
    ('sadness', 'fear', 'love'): 'grief',
    ('sadness', 'surprise', 'joy'): 'regret',
    ('sadness', 'surprise', 'fear'): 'pity',
    ('sadness', 'joy', 'love'): 'yearning',
    ('anger', 'joy', 'love'): 'zeal',
}

# Initialize an empty list to store the result history
history = []

# Function to preprocess text (remove punctuations and convert to lowercase)
def preprocess_text(text):
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

# Function to calculate complex emotions based on rules and probabilities
def calculate_complex_emotion(probabilities):
    # Create a set of emotions that exceed the threshold
    exceeded_threshold = set()
    emotions = ['anger', 'fear', 'sadness', 'surprise', 'joy', 'love']

    for i, prob in enumerate(probabilities):
        if prob > threshold:
            exceeded_threshold.add(emotions[i])

    # Check if any complex emotion rules match the exceeded emotions
    for rule_emotions, complex_emotion in complex_emotions.items():
        if set(rule_emotions).issubset(exceeded_threshold):
            return complex_emotion

    return None  # If no complex emotion rule matches, return None

# Function to predict emotion from text
def predict_emotion(text):
    # Preprocess the input text
    text = preprocess_text(text)

    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(text_sequence, maxlen=50, padding='post', truncating='post')

    # Perform emotion inference
    probabilities = model.predict(padded_sequence)

    # Calculate the complex emotion based on the rules and probabilities
    complex_emotion = calculate_complex_emotion(probabilities[0])

    # Return the complex emotion and probabilities
    return complex_emotion, probabilities[0]

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    global history  # Declare history as a global variable
    random_string = ""  # You can set a random string here if needed

    # Load history from the JSON file
    load_history()

    if request.method == 'POST':
        user_text = request.form['text']
        complex_emotion, probabilities = predict_emotion(user_text)

        # Create a timestamp for the current prediction
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Add the current prediction to the history list
        history.append({
            'text': user_text,
            'emotions': ['anger', 'fear', 'sadness', 'surprise', 'joy', 'love'],  # Define your emotions here
            'probabilities': probabilities.tolist(),
            'inferred_emotion': complex_emotion,
            'timestamp': timestamp
        })

        # Sort the updated history by timestamp
        history = sort_history_by_timestamp(history)

        # Save the updated history to a JSON file
        save_history()

    return render_template('index.html', random_string=random_string, threshold=threshold, history=history)

# Route to generate a random string from "test_text.txt" and update the textarea
@app.route('/random', methods=['POST'])
def random_text():
    with open('test_text.txt', 'r') as file:
        lines = file.readlines()
        random_string = random.choice(lines).strip()

    return render_template('index.html', random_string=random_string, threshold=threshold, history=history)

# Route to set the inference threshold
@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global threshold  # Make sure "threshold" is a global variable
    new_threshold = float(request.form['threshold'])  # Assuming you're sending the threshold value from a form
    threshold = new_threshold
    return redirect('/')  # Redirect back to the home page after setting the threshold

# Route to clear the history
@app.route('/clear', methods=['POST'])
def clear_history():
    history.clear()
    save_history()
    return redirect('/')

# Function to load history from history.json
def load_history():
    global history
    try:
        with open('history.json', 'r') as history_file:
            history = json.load(history_file)
    except FileNotFoundError:
        history = []

# Function to save history to history.json
def save_history():
    with open('history.json', 'w') as history_file:
        json.dump(history, history_file, indent=4)

# Function to sort history by timestamp in descending order
def sort_history_by_timestamp(history):
    return sorted(history, key=lambda x: x['timestamp'], reverse=True)

if __name__ == '__main__':
    app.run(debug=True)
