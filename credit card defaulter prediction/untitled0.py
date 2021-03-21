import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

default = pd.read_csv('TRAIN.csv', index_col="ID")

print(default.head())
#,"\n",default.describe())

default.rename(columns=lambda x: x.lower(), inplace=True)
#Base values :female ,other education,not married
default['grad_school'] = (default['education']==1).astype('int')
default['university'] = (default['education']==2).astype('int') #// features creat korse education r jnno..
default['high_school'] = (default['education']==3).astype('int')
default.drop('education', axis=1, inplace=True)

default['male'] = (default['sex']==1).astype('int')
default.drop('sex', axis=1, inplace=True)

#default['married'] = (default['marraige'] == 1).astype('int')
#default.drop('marraige', axis=1, inplace=True)
#for pay features if the <=0 then it means it was not delayed
pay_features = ['pay_0','pay_2','pay_3','pay_4','pay_5','pay_6']
for p in pay_features:
     default.loc[default[p]<=0, p] = 0
default.rename(columns={'default payment next month':'default'}, inplace=True)
target_name= 'default'
X = default.drop('default' , axis=1)
robust_scaler = RobustScaler()
x = robust_scaler.fit_transform(X)
y= default[target_name]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=123, stratify=y)

def CMatrix(CM, labels=['pay','default']):
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name='TRUE'
    df.columns.name='PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df
metrics= pd.DataFrame(index=['accuracy','precision', 'recall'],
                      columns=['NULL', 'LogisticReg','ClassTree', 'NaiveBayes'])
y_pred_test= np.repeat(y_train.value_counts().idxmax(), y_test.size)
metrics.loc['accuracy','NULL'] = accuracy_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision','NULL'] = precision_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall','NULL'] = recall_score(y_pred=y_pred_test, y_true=y_test)
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
W=CMatrix(CM)
print(W)

#LOGISTIC REGGRESSION

logistic_regression = LogisticRegression(n_jobs=-1, random_state=15)

# Use the training data to train the estimator 
logistic_regression.fit(x_train, y_train) 

# Evaluate the model 
y_pred_test = logistic_regression.predict(x_test)
metrics.loc['accuracy','LogisticReg'] = accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','LogisticReg'] = precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','LogisticReg'] = recall_score(y_pred=y_pred_test,y_true=y_test)
#confusion matrix 
CM = confusion_matrix(y_pred=y_pred_test,y_true=y_test)
Q=CMatrix(CM)
print('logistic regression') 
print(Q)

#DECISION TREES

class_tree = DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10, random_state=10)
# creat an instance of the estimator
class_tree.fit(x_train, y_train)

#evalute the model
y_pred_test = class_tree.predict(x_test)
metrics.loc['accuracy','ClassTree'] = accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','ClassTree'] = precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','ClassTree'] = recall_score(y_pred=y_pred_test,y_true=y_test)
#confusion matrix 
CM = confusion_matrix(y_pred=y_pred_test,y_true=y_test)
A=CMatrix(CM)
print('decision tree') 
print(A)

#GAUSSIANNB


#Create an instance of the estimator 
NBC =  GaussianNB()

#Use the training data to train the estimator 
NBC.fit(x_train,y_train)

#Evaluate the model 
y_pred_test = NBC.predict(x_test) 
metrics.loc['accuracy','NaiveBayes'] = accuracy_score(y_pred=y_pred_test,y_true=y_test)

metrics.loc['precision','NaiveBayes'] = precision_score(y_pred=y_pred_test,y_true=y_test)

metrics.loc['recall','NaiveBayes'] = recall_score(y_pred=y_pred_test,y_true=y_test)

#Confusion Matrix
 
CM = confusion_matrix(y_pred=y_pred_test,y_true=y_test)
G=CMatrix(CM) 
print('naive bayes')
print(G)
100*metrics
fig,ax =plt.subplots(figsize=(8,5))
metrics.plot(kind='barh', ax=ax)
ax.grid();
precision_nb, recall_nb, thresholds_nb = precision_recall_curve(y_true=y_test,probas_pred=NBC.predict_proba(x_test)[:,1])
precision_lr,recall_lr,thresholds_lr = precision_recall_curve(y_true=y_test,probas_pred=logistic_regression.predict_proba(x_test)[:,1])

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(precision_nb, recall_nb, label='NaiveBayes')
ax.plot(precision_lr, recall_lr, label='LogisticReg')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision-Recall Curve')
ax.hlines(y=0.5, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid();
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(thresholds_lr, precision_lr[1:], label='Precision')
ax.plot(thresholds_lr, recall_lr[1:], label='Recall')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Precision, Recall')
ax.set_title('Logistic Regression Classifier: Precision-Recall')
ax.hlines(y=0.6, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid();
y_pred_proba = logistic_regression.predict_proba(x_test)[:,1]
y_pred_test = (y_pred_proba >= 0.2).astype('int')

#Confusion matrix

CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
print("Recall: ",100*recall_score(y_pred=y_pred_test, y_true=y_test))
print("Precision: ",100*precision_score(y_pred=y_pred_test, y_true=y_test))
N=CMatrix(CM)
print('recall & precision')
print(N)
def make_ind_prediction(new_data):
    data = new_data.values.reshape(1, -1)
    data = robust_scaler.transform(data)
    prob = logistic_regression.predict_proba(data)[0][1]
    if prob >= 0.2: 
        return 'Will default'
    else:
        return 'Will pay'
pay = default[default['default']==0]   
pay.head()
from collections import OrderedDict
new_customer = OrderedDict([('limit_bal', 4000),('age', 50),('bill_amt1', 500),
                            ('bill_amt2', 35509),('bill_amt3', 689),('bill_amt4', 0),
                              ('bill_amt5', 0),('bill_amt6', 0),('pay_amt1', 0),('pay_amt2', 0),
                              ('pay_amt3', 0),('pay_amt4', 0),('pay_amt5', 0),('pay_amt6', 0),
                              ('male', 1), ('grad_school', 0),('university',1),('high_school', 0),
                              ('married', 1),('pay_0', -1),('pay_2', -1),('pay_3', -1),
                              ('pay_4', 0),('pay_5', -1),('pay_6', 0)])
 
new_customer = pd.Series(new_customer)
K=make_ind_prediction(new_customer)
#print( "Will Default")
print((K))
