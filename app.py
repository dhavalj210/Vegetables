# app.py
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import json
import random
from flask import Flask, render_template, request,jsonify

app = Flask(__name__)

# Load data from intents.json
with open('intents.json') as f:
    data = json.load(f)

# Initialize stemmer and lists
stemmer = LancasterStemmer()
words = []
labels = []
docs_x = []
docs_y = []

# Process patterns and tags from intents
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        
# Stem and preprocess words
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Create training data and output
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Define bag_of_words and get_response functions
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def get_response(user_message):
    # Load the trained model
    loaded_model = load_model("model.keras")

    # Preprocess user input
    user_input = bag_of_words(user_message, words)

    # Get model prediction
    results = loaded_model.predict(np.array([user_input]))
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            print("Responses:", responses)
            chatbot_response = random.choice(responses)
            return chatbot_response

@app.route('/')
def index():
    return render_template('index.html')

model = load_model("model.keras")
@app.route('/chat', methods=['POST'])


def chat():
    user_message = request.form['user_message']
    bot_response = get_response(user_message)
    return jsonify({'bot_response': bot_response})



if __name__ == '__main__':
    app.run(debug=True)