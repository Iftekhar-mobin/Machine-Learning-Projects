import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt

#LOADING FILES INTO PANDAS DATAFRAME

df = pd.read_csv("/home/likewise-open/BRACU/14101161/Desktop/Churn.csv")

df.drop(df.columns[[0]], axis=1, inplace=True)
#THERE WAS EXTRA COLUMN WHICH IS DROPPED
#print(df.head())


#print(df.describe())

#AS IT IS NOT PART OF OUR ANALYSIS ,SO IT IS DROPPED
df = df.drop(["Phone", "Area Code", "State"], axis=1)
#CHURN IS OUR PREDICTIVE VARIAVLE ,so it is dropped
features = df.drop(["Churn"], axis=1).columns
#we are spliting the dataset into 75% training set and 25% test set
df_train, df_test = train_test_split(df, test_size=0.25)

#initializing random forrest classifier
clf = RandomForestClassifier(n_estimators=30)

#training the classifier
clf.fit(df_train[features], df_train["Churn"])

# Making predictions
predictions = clf.predict(df_test[features])
#probabilities of the features
probs = clf.predict_proba(df_test[features])
print("value of prediction ",predictions)
#accuracy of our model
score = clf.score(df_test[features], df_test["Churn"])
print("Accuracy: ", score)

from sklearn.metrics import confusion_matrix
'''cm = confusion_matrix(df_test, predictions)
//print(cm)
'''
#making a confusion matrix to look as true positive ,true negative ,false positive and false negative
confusion_matrix = pd.DataFrame(
    confusion_matrix(df_test["Churn"], predictions),
    columns=["Predicted False", "Predicted True"],
    index=["Actual False", "Actual True"]
)

print(confusion_matrix)

#ploting ROC curve
fpr, tpr, threshold = roc_curve(df_test["Churn"], probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


fig = plt.figure(figsize=(20, 18))
ax = fig.add_subplot(111)


#finding which feature has more importance in the prediction
df_f = pd.DataFrame(clf.feature_importances_, columns=["importance"])
df_f["labels"] = features
df_f.sort_values("importance", inplace=True, ascending=False)
print(df_f.head(5))
import numpy as np
index = np.arange(len(clf.feature_importances_))
bar_width = 0.5
rects = plt.barh(index , df_f["importance"], bar_width, alpha=0.4, color='b', label='Main')
plt.yticks(index, df_f["labels"])
plt.show()


df_test["prob_true"] = probs[:, 1]
df_risky = df_test[df_test["prob_true"] > 0.9]
print(df_risky.head(5)[["prob_true"]])