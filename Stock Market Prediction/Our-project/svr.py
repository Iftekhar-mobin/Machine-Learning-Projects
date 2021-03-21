from sklearn.svm import SVR
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
	dates = np.reshape(dates,(len(dates), 1)) 

	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) 
	svr_lin = SVR(kernel= 'linear', C= 1e3)
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf.fit(dates, prices) 
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)

	plt.scatter(dates, prices, color= 'green', label= 'Data') 
	plt.plot(dates, svr_rbf.predict(dates), color= 'orange', label= 'RBF model') 
	plt.plot(dates,svr_lin.predict(dates), color= 'red', label= 'Linear model') 
	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') 
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Stock Market Prediction')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

my_data('saad.csv') 
print ("Dates- ", dates)
print ("Prices- ", prices)

my_price = my_price(dates, prices, 29)  
print ("\nThe stock open price for 29th Feb is:")
print ("RBF kernel: ", str(my_price[0]))
print ("Linear kernel: ", str(my_price[1]))
print ("Polynomial kernel: ", str(my_price[2]))
	 
