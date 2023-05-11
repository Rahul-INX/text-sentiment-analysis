from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts([''])
model = tf.keras.models.load_model(r"E:\College\Sem_6\Mini_Project\emotion_model_trained_2.h5")

@app.route('/')
def home():
    return render_template('E:\College\Sem 6\Mini Project\index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    preprocessed_input = tokenizer.texts_to_sequences([text])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(preprocessed_input, truncating='post', maxlen=50, padding='post')
    predicted_label = np.argmax(model.predict(padded_input), axis=-1)[0]
    predicted_emotion = {0: 'anger', 1: 'fear', 2: 'sadness', 3: 'surprise', 4: 'joy', 5: 'love'}[predicted_label]
    return render_template('Sem 6/Mini Project/result.html', emotion=predicted_emotion)


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
