import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#read input
dataset = pd.read_csv('422.csv', header=0, index_col=0)
#print(dataset.head(),"\n",dataset.describe())

#taking moving averages of 28 days and 260 days sliding window method
real = dataset.Close
mv1 = dataset.Close.rolling(28).mean()
mv2 = dataset.Close.rolling(260).mean()
mv3 = dataset.Close.rolling(500).mean()

# convert series to frame
mv1 = mv1.to_frame()
mv2 = mv2.to_frame()
mv3 = mv3.to_frame()
diff = mv1 - mv2
diff1 = mv2 - mv3


# adding new header in the column
mv1.columns = ['MA1']
mv2.columns = ['MA2']
mv3.columns = ['MA3']
diff.columns= ['diff']
diff1.columns=['diff1']

# Adding all the columns together
frames = [real, mv1, mv2, mv3, diff, diff1]
total = pd.concat(frames, axis=1)

# check whether it contains any Nan value
print(total.head())

#drop row if there is any value zero within a row
final = total.dropna(axis=0, how="any")

# check whether it contains any Nan value after applying dropna
print(final.head())

# see the results in charts
#final[['Close','MA1','MA2']].plot(figsize=(20,10))
#plt.show()

# below is the logic when to start selling and when to buy
X = 0.05
final['dec'] = np.where(final['diff'] > X, 1, 0)
final['dec'] = np.where(final['diff'] < X, -1, final['dec'])
final['dec1']= np.where(final['diff1'] > X, 1,0)
final['dec1']= np.where(final['diff1'] < X, -1,final['dec1'])


print("Total number of buy and Sell Signal \n",final['dec'].value_counts())
print("Total number of buy and Sell Signal \n",final['dec1'].value_counts())

# put all together in one chart to compare the accuracy of our strategy
plt.figure(figsize=(6, 4))
plt.subplot(2, 1, 1)
final[['Close','MA1','MA2','MA3']].plot(ax=plt.gca())
plt.subplot(2, 1, 2)
final['dec'].plot()
plt.xticks(())
plt.yticks(())
plt.text(0.1, 0.5, 'Buy Sell Decision(short term)', ha='center', va='center',size=10, alpha=.5)
plt.tight_layout()
plt.show()

plt.subplot(2, 1, 2)
final['dec1'].plot()
plt.xticks(())
plt.yticks(())
plt.text(0.1, 0.5, 'Buy Sell Decision(Long term)', ha='center', va='center',size=10, alpha=.5)
plt.tight_layout()
plt.show()
