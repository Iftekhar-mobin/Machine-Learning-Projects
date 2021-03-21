from sklearn import linear_model
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

dates = []
prices = []

def my_data(myfile):
	with open(myfile, 'r') as csvfile:
		saad_csv = csv.reader(csvfile)
		next(saad_csv) 
		saad_time = time.time()
		for row in saad_csv:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
		print("---- my code's time : %s in %s seconds ---" % (myfile,time.time() - saad_time))
	return

def my_price(dates, prices, x):
	dates = np.reshape(dates, (len(dates),1)) 
	prices = np.reshape(prices, (len(prices),1))
	
	saad_linear = linear_model.LinearRegression() 
	saad_linear.fit(dates, prices)
	
	plt.scatter(dates, prices, color= 'green', label= 'Data')  
	plt.plot(dates, saad_linear.predict(dates), color= 'red', label= 'Linear model') 
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Stock Market Prediction wiith Linear Regression')
	plt.legend()
	plt.show()
	
	return saad_linear.predict(x)[0][0], saad_linear.coef_[0][0], saad_linear.intercept_[0]

my_data('saad.csv') 
print ("Dates- ", dates)
print ("Prices- ", prices)

my_price, coefficient, constant = my_price(dates, prices, 29)  
print ("\nThe stock open price for 29th Feb is: ", str(my_price))
print ("The regression coefficient =", str(coefficient), ", and the constant = ", str(constant))
print ("the relationship equation between dates and prices is: price = ", str(coefficient), "* date + ", str(constant))
