import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
df = pd.read_csv('orders/orders.csv')
product=pd.read_csv('products/products.csv')
product=product.head(n=49688//4)
final= pd.merge(df, product, on="product_id")
orders=final.values.tolist()
transactions={}
for i in range(len(orders)):
    x=orders[i][0]
    if x in transactions:
        transactions[x].append(orders[i][4])
    else:
        transactions[x]=[orders[i][4]]
text=[]
for i in transactions:
    if len(transactions[i])!=1:
        text.append(";".join(transactions[i]))
text = "|".join(text)
def replace(text):
    if ' ' in text: 
        text = text.replace(' ' , '_')
        text = text.replace('-','_')
        text = text.replace(',','_')
    return text
t=replace(text)
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
tokenizer = Tokenizer(filters=';|',lower=True)

tokenizer.fit_on_texts([t])
vocabulary_size = len(tokenizer.word_index) + 1
print('Unique items: %d' % vocabulary_size)
sequences = list()
for line in text.split('|'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    encoded=encoded[:5]
    sequences.append(encoded)
print('Total Sequences: %d' % len(sequences))
pickle.dump(tokenizer, open("token.pickle", "wb"))
max_len = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print('Max Sequence Length: %d' % max_len)
sequences = np.array(sequences)
from sklearn.model_selection import train_test_split
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocabulary_size)
x_train, x_test, y_train, y_test = train_test_split(X[:131209//2], y[:131209//2], test_size=0.2)

from keras import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras.callbacks import EarlyStopping
model = Sequential()
model.add(Embedding(vocabulary_size, 4, input_length=max_len - 1))
model.add(LSTM(16))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(vocabulary_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        es_callback = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
		#h = model.fit(X, y, validation_split=0.2, verbose=1, epochs=5000, callbacks=[es_callback])
        r = model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=0,epochs=5000, callbacks=[es_callback])
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)