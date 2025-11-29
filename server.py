"""
AI Chatbot Server

This module implements a Flask-based web server for an AI chatbot.
It handles:
1.  Loading and preprocessing training data (intents and patterns).
2.  Training or loading a Deep Neural Network (DNN) model using TFLearn.
3.  Providing API endpoints for the chatbot to interact with users via web interface or API calls.
"""

import nltk
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
try:
    import tensorflow
except Warning:
    pass

import random
import json
import pickle

from flask import Flask
from flask import *

app = Flask(__name__)

# Load intents data
with open('data_to_train.json', 'r') as f:
    data = json.loads(f.read())


# Load or Process Data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Stemming and sorting
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Reset graph and define model
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    """
    Converts a sentence into a bag-of-words array.

    The function tokenizes the sentence, stems the words, and checks their existence
    in the known vocabulary (`words`). It returns a one-hot encoded array indicating
    which words from the vocabulary are present in the sentence.

    Args:
        s (str): The input sentence to be processed.
        words (list): A list of known vocabulary words (stemmed and sorted).

    Returns:
        numpy.ndarray: A numpy array representing the bag of words. The array contains
        1s at indices where the word exists in the sentence and 0s otherwise.
    """
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    """
    Starts a command-line interface chat session with the bot.

    This function runs a loop where the user can input messages via the console.
    The bot predicts the intent of the message and returns a random response
    associated with that intent. Typing "quit" exits the loop.

    Returns:
        None
    """
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


def chat_flask(inp):
    """
    Processes a chat message and returns a response suitable for the Flask app.

    This function predicts the intent of the input message using the trained model
    and selects a random response from the identified intent's list of responses.

    Args:
        inp (str): The input message from the user.

    Returns:
        str: A response string from the chatbot. If the input is "quit", it returns None.
    """
    if inp.lower() == "quit":
        return

    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)

# Initialize chat history
allchats = [
    {'id':'bot',
    'msg': 'hi'},
    {'id':'user',
    'msg': 'hello'}
]

@app.route("/", methods=['GET', 'POST'])
def Hello():
    """
    Handles requests to the root URL ("/") to display the chat interface.

    - GET: Renders the `index.html` template with the current chat history.
    - POST: Processes the user's message submitted via the form, generates a bot response,
      updates the chat history, and re-renders the page.

    Returns:
        flask.Response: The rendered HTML template for the chat interface.
    """
    if request.method == 'POST':
        msg = request.form['msg']
        res = chat_flask(msg)
        allchats.append({'id':'user', 'msg':msg})
        allchats.append({'id':'bot', 'msg':res})
        return render_template('index.html', chats=allchats)
    return render_template('index.html', chats=allchats)

@app.route("/response", methods=["GET", "POST"])
def getId():
    """
    API endpoint to get a chatbot response via POST request.

    This endpoint expects a POST request with form data containing a 'msg' field.
    It returns the bot's response as a plain string.

    Returns:
        str: The chatbot's response if the method is POST.
        str: "nooooo" if the method is GET (indicating incorrect usage for getting a response).
    """
    if request.method == "POST":
        return chat_flask(request.form['msg'])
    else:
        return "nooooo"
        
@app.route("/response/<string:msg>", methods=['GET'])
def getApiResponse(msg):
    """
    API endpoint to get a chatbot response via GET request parameters.

    This endpoint allows passing the message directly in the URL path.

    Args:
        msg (str): The user's message captured from the URL.

    Returns:
        str: The chatbot's response.
    """
    return chat_flask(msg)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
