from flask import Flask, render_template, request, redirect
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import datetime

# Set the minimum number of entries required for depression calculation
MIN_ENTRIES = 10


app = Flask(__name__)
app.config['THRESHOLD'] = 50

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the emotion trained model
model = tf.keras.models.load_model('emotion_model_trained.h5')

# Create an empty list to store the input and result history
history = []

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

def save_history(text, emotion):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Append the input, emotion, and timestamp to the history list
    history.insert(0, {'text': text, 'emotion': emotion, 'timestamp': timestamp})

    # Save the updated history to a file
    with open('history.txt', 'a') as f:
        f.write(f"{text}\t{emotion}\t{timestamp}\n")

def analyze_history(window_size):
    current_time = datetime.datetime.now()
    start_time = current_time - window_size

    # Filter history within the window size
    filtered_history = [entry for entry in history if start_time <= datetime.datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') <= current_time]

    # Calculate the emotion frequencies
    emotion_counts = {}
    for entry in filtered_history:
        emotion = entry['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Calculate the average emotion
    total_entries = len(filtered_history)
    average_emotion = {emotion: count / total_entries * 100 for emotion, count in emotion_counts.items()}

    return average_emotion

def check_depression_warning(min_entries):
    if len(history) < min_entries:
        return False

    # Calculate average emotions for different window sizes
    one_hour_average = analyze_history(datetime.timedelta(hours=1))
    one_day_average = analyze_history(datetime.timedelta(days=1))
    one_month_average = analyze_history(datetime.timedelta(days=30))

    # Check if specific emotions exceed the threshold
    threshold = app.config['THRESHOLD']  # Get threshold from app configuration
    threshold_emotions = ['anger', 'sadness', 'fear', 'surprise']
    depression_warning = False

    for emotion in threshold_emotions:
        if (
            one_hour_average.get(emotion, 0) > threshold or
            one_day_average.get(emotion, 0) > threshold or
            one_month_average.get(emotion, 0) > threshold
        ):
            depression_warning = True
            break

    return depression_warning

def load_history():
    try:
        with open('history.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                entry = line.strip().split('\t')
                history.append({'text': entry[0], 'emotion': entry[1], 'timestamp': entry[2]})
    except FileNotFoundError:
        # Create the history file if it doesn't exist
        open('history.txt', 'w').close()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        emotion = predict_emotion(text)
        save_history(text, emotion)
        return redirect('/')
    else:
        emotion = None
        threshold = app.config['THRESHOLD']
        depression_warning = check_depression_warning(MIN_ENTRIES)
        return render_template('index.html', emotion=emotion, history=history, depression_warning=depression_warning, threshold=threshold)


@app.route('/clear', methods=['POST'])
def clear_history():
    history.clear()
    with open('history.txt', 'w') as f:
        pass  # Clear the history file
    return redirect('/')



@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    threshold = int(request.form['threshold'])
    app.config['THRESHOLD'] = threshold
    return redirect('/')


if __name__ == '__main__':
    load_history()
    app.run(debug=True)

