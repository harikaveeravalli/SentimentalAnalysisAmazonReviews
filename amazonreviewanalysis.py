import re
#import nltk
#import tensorflow as tf
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix


#with open('/Users/harika/Downloads/reviews.csv', 'r') as myfile:
#    reader = myfile.read().splitlines()

dataframe1=pd.read_csv('/Users/harika/Downloads/reviews.csv','|',nrows=100000)
#print dataframe1.label
testframe=dataframe1.iloc[4::5, :]
#print testframe
trainframe=dataframe1.drop(testframe.index)
trainframe.index=range(len(trainframe))
testframe.index=range(len(testframe))

#testdata=reader[4:len(reader)+1:5]
#traindata=[value for value in reader if value not in testdata]
#convert the dataframe values to lower case

#testframe["text"]=testframe["text"].astype(str).str.lower()
#trainframe["text"]=trainframe["text"].astype(str).str.lower()

#for val in traindata:
#    strsentence=val.split("|")
#    label.append(strsentence[0])

#    reviewText.append(strsentence[1].lower())

#splitting test data
#for val2 in testdata:
#    strsentence2=val2.split("|")
#   label2.append(strsentence2[0])
#    reviewTestset.append(strsentence2[1].lower())
#print label
#convert the list of labels into 0's and 1's
trainframe=trainframe.replace("positive",1)
trainframe=trainframe.replace("negative",0)
testframe=testframe.replace("positive",1)
testframe=testframe.replace("negative",0)
print("completed replacing")
#print label
#convert positives, negatives as o's and 1's
#for word in label2:
#    if word == 'positive':
#        label2[label2.index(word)]=1
#    else:
#        label2[label2.index(word)]=0


#regular expression to remove punctutations from the string and make a new
#for string1 in reviewText:
#    print("entered the loop")
#    string1=re.sub(r'[^ a-zA-Z]',' ',string1)
#    print string1
#    reviewTextPunct.append(string1)
for counter, row in trainframe.iterrows():
    row["text"]= re.sub(r'[^a-zA-Z]',' ',row["text"])
print("completed removing punctutations")
stop_words = set(stopwords.words('english'))
#removing stop words from the review text
#print stop_words

#unique_words=list(set(" ".join(reviewTextPunct).split(" ")))
#unique_words = [w for w in unique_words if w not in stop_words]
unique_words=list(trainframe["text"].str.split(' ', expand=True).stack().unique())
print("created unique words")
unique_words = [w for w in unique_words if w not in stop_words]
print("removed stop words")
matrix=[[0 for i in range(len(unique_words))] for j in range(len(trainframe))]
#print unique_words
print("completed unique words")
for i,row in trainframe.iterrows():
    for word in unique_words:
        if word in row["text"]:
            matrix[i][unique_words.index(word)]=1

#print matrix
print "completed test matrix"
#matrix for test data
testMatrix=[[0 for i in range(len(unique_words))] for j in range(len(testframe))]

for i,row in testframe.iterrows():
    for word in unique_words:
        if word in row["text"]:
            testMatrix[i][unique_words.index(word)]=1
#print testMatrix
print "completed test matrix"

#preprocessing the data
scaler=StandardScaler()
scaler.fit(matrix)
matrix=scaler.transform(matrix)
testMatrix=scaler.transform(testMatrix)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf=MLPClassifier(hidden_layer_sizes=(5,5,5))

clf.fit(matrix, trainframe["label"])
print("completed fit")
#MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#       beta_1=0.9, beta_2=0.999, early_stopping=False,
#       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#       warm_start=False)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5,5,5), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
print("completed classifier call")
predictions=clf.predict(testMatrix)

#print(clf.predict(testMatrix))

#print predictions

print("completed prediction")
print(confusion_matrix(testframe["label"],predictions))
print(classification_report(testframe["label"],predictions))