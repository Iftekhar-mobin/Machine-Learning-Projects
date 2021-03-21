import csv
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pandas import read_csv
import re

data100 = pd.read_csv('AudioDec.csv', dtype={'ID': object})
data100['text'] = data100['text'].apply(
    lambda y: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in y]))
print("\n => Non ASCII Characters Removed")
data100.to_csv('GameplayJAN2.csv', encoding='utf-8')
# ============================================= Lemmatizing Started ============================================== #


df = pd.DataFrame.from_csv('GameplayJAN2.csv')
myList = df["text"].values

print("\n========== Cleaning Started ==========")


def get_pos_tag(tag):
    if tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('J'):
        return 'a'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'


lemmatizer = WordNetLemmatizer()
with open('cleanedFile.csv', 'w+',
          newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for doc in myList:
        tok_doc = nltk.word_tokenize(doc)
        pos_tag_doc = nltk.pos_tag(tok_doc)
        lemmatized = []
        for i in range(len(tok_doc)):
            tag = get_pos_tag(pos_tag_doc[i][1])
            if tag == 'r':
                if tok_doc[i].endswith('ly'):
                    temp = tok_doc[i].replace("ly", "")
                else:
                    temp = tok_doc[i].replace("", "")
            else:
                temp = lemmatizer.lemmatize(tok_doc[i], pos=tag)
            lemmatized.append(temp)
        lemmatized = " ".join(lemmatized)
        wr.writerow([lemmatized])
    print("\n => Lemmatization Done")

# ============================================= Lemmatizing Ended ============================================== #



# ============================================= Regex Cleaning ============================================== #

data = pd.read_csv('temp.csv', dtype={'ID': object})

data.dropna()
print("\n => NAN values deleted")

data['text'] = data['text'].str.replace('http : \S+|www.\S+', '', case=False)
data['text'] = data['text'].str.replace('https : \S+|www.\S+', '', case=False)
print("\n => Hyperlinks Removed ")

data['text'] = data['text'].str.replace('@ \S+|# \S+', '', case=False)
data['text'] = data['text'].str.replace('http : \S+|www.\S+', '', case=False)
data['text'] = data['text'].str.replace('twitter\S+', '', case=False)
print("\n => Tweeter Tag Removing Done")

data['text'] = data['text'].apply(
    lambda y: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in y]))
print("\n => Non ASCII Characters Removed")

data['text'] = data['text'].str.replace(r'<[^>]+>', '', case=False)
data['text'] = data['text'].str.replace('=|%|-|<|>|`', '', case=False)
print("\n => HTML TAG Removing Done")

data['text'] = data['text'].str.replace(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', case=False)
print("\n => Numbers Removing Done")

data['text'] = data['text'].str.replace(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", '', case=False)
print("\n => Hash-Tag Removing Done")

data['text'] = data['text'].str.replace('\!|\?|\@|\.|&|\/|\:|\;|\(|\)', '', case=False)
print("\n => Punctuations Removing Done")

data['text'] = data['text'].str.replace(r'[\'\",]*', '', case=False)
print("\n => Extra Commas Removing Done")

data['text'] = data['text'].str.replace('\'s', ' is', case=False)
print("\n => Apostrophe Removing Done")

data['text'] = data['text'].str.replace('snapchat|facebook|twitter|instagram|linkdin|google', '', case=False)
print("\n => Social Tag Removing Done")

# data.to_csv('cleanedFile.csv', encoding='utf-8')


# ============================================= Regex Cleaning Ended ============================================== #


# ============================================= Stop Words Removing ============================================== #

data['temp'] = data['text'].str.lower().str.split()
data['text'] = data['temp'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])
print("\n => Stop Words Removing Done")

# ============================================= Stop Words Removing ============================================== #



del data['temp']

data.to_csv('AudioDecCleaned.csv', encoding='utf-8')

print("\n========== Cleaning Ended ==========")
print("\n Your Cleaned File ('cleanedFile.csv') Has Generated")

# ============================================= Final Stage ============================================== #

