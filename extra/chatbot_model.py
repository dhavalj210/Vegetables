import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import json
import random

# Initialize stemmer and lists
stemmer = LancasterStemmer()
words = []
labels = []
docs_x = []
docs_y = []

# Load data from intents.json
with open('intents.json') as f:
    data = json.load(f)

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

# Convert lists to numpy arrays
training = np.array(training)
output = np.array(output)

# Create and train the model
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training, output, epochs=1000, batch_size=8, verbose=1)

# Save the model using tf.keras.models.save_model
model.save("model.keras")

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
            chatbot_response = random.choice(responses)
            return chatbot_response
