
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

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


text = input('enter your ')
emotion = predict_emotion(text)
print("Predicted Emotion:", emotion)

