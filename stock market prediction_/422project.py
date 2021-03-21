import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing training set
training_set=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=training_set.iloc[:,1:2].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
training_set=sc.fit_transform(training_set)

#getting the inputs and outputs
x_train=training_set[0:1257]
y_train=training_set[1:1258]

#reshaping
x_train=np.reshape(x_train, (1257, 1, 1))

#building the RNN

#importing the keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#initialising the RNN

regressor=Sequential()

#adding the input layer and the LSTM layer 

regressor.add(LSTM(units=4, activation= 'sigmoid', input_shape=(None,1)))

#adding the output layer

regressor.add(Dense(units=1))

#compiling the RNN
regressor.compile(optimizer= 'adam', loss='mean_squared_error')

#fitting the RNN to the training set
regressor.fit(x_train, y_train, batch_size=32, epochs=200)

#making prediction and visualising the result

#getting real stock price

test_set=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_set.iloc[:,1:2].values

#getting the prediction
inputs=real_stock_price
inputs=sc.transform(inputs)
inputs=np.reshape(inputs, (20,1,1)) 

predicted_stock_price=regressor.predict(inputs)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)


#visualising the result

plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#getting real stock price
real_stock_price_train=pd.read_csv('Google_stock_price_Train.csv')
real_stock_price_train=real_stock_price_train.iloc[:,1:2].values

#predicting stock price
predicted_stock_price_train=regressor.predict(x_train)
predicted_stock_price_train=sc.inverse_transform(predicted_stock_price_train)
#visualising result

plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#evaluate

import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))









