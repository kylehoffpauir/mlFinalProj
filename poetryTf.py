import tensorflow as tf

import numpy as np
import os
import time
import pandas as pd

data = pd.read_csv('PoetryFoundationData.csv', encoding='utf-8-sig')
print(data.info())
# https://github.com/tensorflow/text/blob/master/docs/tutorials/text_generation.ipynb
# https://www.te
# nsorflow.org/text/tutorials/text_generation
import re
lyrics = data['Poem']

rawtext = ""
for l in lyrics:
    for character in l:
        line = ""
        match = re.search("[a-zA-Z0-9\n ]",character)
        if match:
            line += character
        rawtext += line
# print(rawtext)
maxlen = 60 #extract sequences of length 60
step = 3
sentences = []	#holds extracted sequences
next_chars = [] #holds the targets

for i in range(0, len(rawtext)-maxlen, step):
    sentences.append(rawtext[i:i+maxlen])
    next_chars.append(rawtext[i+maxlen])

#VECTORIZATION
chars = sorted(set(rawtext))
vocab_len = len(chars)
char_indices = dict((char, chars.index(char)) for char in chars)
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
# uses indices
# x shape has three parts: SEQUENCES, length of the sequences, output characters
# y shape is the which character is expected, boolean array off all false and one true to indicate which char
print(char_indices)

from tensorflow.keras import layers
model = tf.keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation="softmax"))
model.compile(loss="categorical_crossentropy", run_eagerly=True, optimizer="adam")

# preds is output neurons, probabilities of which character should be next
# temperature - 1.0 take highest probability and 0.0 more random
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



import random
import sys
for epoch in range(0, 60):
    print("Epoch ", epoch)
    f = open("phil_epoch/epoch_{}.txt".format(epoch), "w")
    f.write("Epoch {}\n".format(epoch))
    # fit the text to the model every epoch
    model.fit(x,y, batch_size=128)
    start_index = random.randint(0, len(rawtext) - maxlen - 1)
    generated_text = rawtext[start_index: start_index + maxlen]
    generated_text = generated_text.replace("_", "").lower()
    print('--- Generating with seed: "' + generated_text + '"')
    f.write("Seed {}\n".format(generated_text))
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        f.write("Temperature {}\n".format(temperature))
        #print(generated_text)
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
            f.write(next_char)
    f.close()

