import  pandas as pd
from gensim import similarities,models
import random
import  re
import pickle
import  os

trainData, trainLabel, testData, testLabel=pickle.load(open('./data.bin', 'rb'))

#使用深度神经网络
from sklearn.preprocessing import Normalizer
trainData=Normalizer().fit_transform(trainData)
testData=Normalizer().fit_transform(testData)

from keras.models import Sequential
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.layers import Dense, Dropout,Activation
from keras.layers import Embedding,Conv2D,MaxPooling2D, Flatten
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping
from keras.layers import LSTM
from keras.utils import np_utils
import keras
import  numpy as np


model = Sequential()
print(np.array(trainData).shape)
#LSTM
#model.add(Embedding(100, output_dim=256))
#model.add(LSTM(100))
#model.add(Dropout(0.2))
#model.add(Dense(1, activation='softmax'))

#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])



#model.fit(np.array(trainData), np.array(trainLabel), epochs = 2, batch_size=256, verbose=1,
#        )

#CNN
trainData = np.array(trainData).reshape(-1, 1,10,10)
testData = np.array(testData).reshape(-1, 1, 10,10)
trainData = np.array(trainData).astype('float32')
testData = np.array(testData).astype('float32')
trainData /=31
testData /=31

trainLabel = np_utils.to_categorical(trainLabel, 2)
testLabel = np_utils.to_categorical(testLabel, 2)

model.add(Conv2D(28, (1,1), activation='relu', input_shape=(1,10,10)))
model.add(Conv2D(28, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(np.array(trainData), np.array(trainLabel), 
          batch_size=32, epochs=20, verbose=1)

score, acc = model.evaluate(np.array(testData), np.array(testLabel), batch_size=100)
print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)


