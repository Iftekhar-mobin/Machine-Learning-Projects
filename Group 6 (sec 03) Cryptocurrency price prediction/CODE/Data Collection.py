import json
import numpy as np
import os
import pandas as pd
import urllib.request

# connect to poloniex's API
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1356998100&end=9999999999&period=300'

# parse json returned from the API to Pandas DF
openUrl = urllib.request.urlopen(url)
r = openUrl.read()
openUrl.close()
d = json.loads(r.decode())
df = pd.DataFrame(d)

original_columns=[u'close', u'date', u'high', u'low', u'open']
new_columns = ['Close','Timestamp','High','Low','Open']
df = df.loc[:,original_columns]
df.columns = new_columns
df.to_csv('E:/Movie/Python/python codes/CSE422/TryNumberOne/data/ethereum2015to2017.csv',index=None)