import nltk
nltk.download('punkt')
nltk.download('wordnet')

import pickle
import numpy as np
import json
import random
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from nltk.stem import WordNetLemmatizer


word_lemmatizer = WordNetLemmatizer()

# Create application
app = Flask(__name__)


# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')


# Load Model
model = load_model('covid_chatbot_model.h5', compile=False)

intents = json.loads(open('intents.json').read())
words_data = pickle.load(open('words_mod.pkl', 'rb'))
classes_data = pickle.load(open('classes_mod.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [word_lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words_data, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words_data)
    for s in sentence_words:
        for i, w in enumerate(words_data):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words_data, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes_data[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# Bind predict function to URL
@app.route('/send', methods=['POST'])
def send():
    msg = str(request.form['name_input'])
    print("User input msg: " + msg)
    if msg != '':
        res = chatbot_response(msg)
        print("Chatbot response msg: " + res)
        return render_template("index.html", str_msg='Covid Bot:  {}'.format(res))


if __name__ == '__main__':
    # Run the application
    app.run()
