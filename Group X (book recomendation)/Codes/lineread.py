filepath="little.txt" 
import NaiveBayesClassifier as s
with open(filepath) as fp:
    line=fp.readline()
    cnt=1
    while line:
        sen=s.sentiment(line)
        print("Line {}:{}".format(cnt,line.strip()))
        print(sen)
        line=fp.readline()
        cnt+=1
