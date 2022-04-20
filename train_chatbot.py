import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

words_mod=[]
classes_mod = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words_mod.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes_mod:
            classes_mod.append(intent['tag'])
            
# lemmatize, lower each word and remove duplicates
words_mod = [lemmatizer.lemmatize(w.lower()) for w in words_mod if w not in ignore_words]
words_mod = sorted(list(set(words_mod)))
# sort classes
classes_mod = sorted(list(set(classes_mod)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes_mod), "classes", classes_mod)
# words = all words, vocabulary
print (len(words_mod), "unique lemmatized words", words_mod)

pickle.dump(words_mod,open('words_mod.pkl','wb'))
pickle.dump(classes_mod,open('classes_mod.pkl','wb'))

# create our training data
training_SET = []
# create an empty array for our output
output_empty = [0] * len(classes_mod)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words_mod:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes_mod.index(doc[1])] = 1

    training_SET.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training_SET)
training_det = np.array(training_SET)
# creating train and test lists. X - patterns, Y - intents
train_X = list(training_det[:,0])
train_Y = list(training_det[:,1])
print("Training data made")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_X), np.array(train_Y), epochs=200, batch_size=5, verbose=1)
model.save('covid_chatbot_model.h5', hist)