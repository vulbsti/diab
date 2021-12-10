import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('pimaindiansdiabetescsv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
# Load data
myData='pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myData,names=names)
print(myData.shape)
myData = myData.values
# Normalize
Scaler = MinMaxScaler(feature_range = (0,1))
myData = Scaler.fit_transform(myData)
myData
#Split

X=myData[:,0:8]
y=myData[:,8]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
# model

model = Sequential()
model.add(Dense(12,input_dim = 8,activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
# compile

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# fit

model.fit(X_train, y_train, epochs = 150, batch_size = 10)
