# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:59:52 2020

@author: JOGESH MISHRA
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout;   


data = pd.read_csv("gender_data.csv")

data.columns = ["name","gender"]

data["name_length"] = [len(str(i))for i in data["name"]]

np.where(data.isnull())

data = data.dropna()

np.where(data.isnull())

temp_data = data[(data['name_length']>=2)]

temp_data.groupby('gender')['name'].count()

names = data['name']

gender = data['gender']

vocab = set(" ".join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)

char_index = dict((c,i) for i,c in enumerate(vocab))

mask = np.random.rand(len(temp_data))<0.8
training_set = temp_data[mask==True]
test_set = temp_data[mask==False]

X_train=[]

y_train=[]

def set_flag(i):
    tmp = np.zeros(39);
    tmp[i] = 1
    return(tmp)

trunc_train_name = [str(i)[0:30] for i in training_set['name']]

for i in trunc_train_name :
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,30-len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    X_train.append(tmp);

for i in training_set['gender']:
   if i == 'm':
       y_train.append([1,0])
   else:
       y_train.append([0,1])


np.asarray(X_train).shape

np.asarray(y_train).shape


X_train, y_train = np.array(X_train),np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],len(vocab)))

classifier = Sequential()
classifier.add(LSTM(units=512, return_sequences=True,input_shape=(X_train.shape[1],len(vocab))))
classifier.add(LSTM(units=512,return_sequences=False))
#classifier.add(LSTM(units=64,return_sequences=True))
#classifier.add(LSTM(units=64,return_sequences=False))
classifier.add(Dense(units=2, activation='softmax'))
classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics = ['accuracy'])

#classifier.fit(X_train,y_train,epochs=10,batch_size=100)

X_test =[]
y_test =[]

trunc_test_name = [str(i)[0:30] for i in test_set['name']]
for i in trunc_test_name:
    tmp = [set_flag(char_index[j])for j in str(i)]
    for k in range(0, 30-len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    X_test.append(tmp)

for i in test_set['gender']:
    if i=='m':
        y_test.append([1,0])
    else:
        y_test.append([0,1])

X_test = np.array(X_test)
X_test.shape

y_test = np.array(y_test)
y_test.shape

classifier.fit(X_train,y_train,batch_size=1000,epochs=10, validation_data=(X_test,y_test))


score, accuracy = classifier.evaluate(X_test,y_test)
print('Test Score = ',score)
print('Test Accuracy = ',accuracy)


############################ Preddict any random Value ###############################

name = ["jagat jyoti mishra"]
input_data = []

for i in name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0, 30-len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    input_data.append(tmp)

input_data = np.array(input_data)

classifier.predict(input_data)
