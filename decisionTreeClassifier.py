import tensorflow as tf
import numpy as np
import re
import time
import pandas as pd
from sklearn import tree
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
import pickle
ts = time.time()
dataframe1 = pd.read_csv('/Users/harika/Downloads/reviews.csv', sep='|')
#dataframe1 = pd.read_csv('/Users/harika/PycharmProjects/sentimentalanalysis/reviews.csv', sep='|')
testframe = dataframe1.iloc[4::5, :].copy()
#testframe = dataframe1.iloc[3].copy()
trainframe = dataframe1.drop(testframe.index)
trainframe.index = range(len(trainframe))
testframe.index = range(len(testframe))


def wordsless2(Sentence):

    Sentence = re.sub(r'\W*\b\w{1,2}\b', '', Sentence)
    #print "remove words"
    #print Sentence

    return Sentence



def removepunct(sentence):

    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    return sentence


# extracting stopwords from the nltk
stop_words = set(stopwords.words('english'))
#remove stop words from train and test data

def remove_stopwords(Sentence):
    Sentence = ' '.join(word for word in Sentence.split() if word not in stop_words)
    #print Sentence
    return Sentence


def lemmatizewords(Sentence):
    lemmatizer = WordNetLemmatizer()

    Sentence = ' '.join(lemmatizer.lemmatize(word1, pos='v') for word1 in Sentence.split())
    #print Sentence
    return Sentence


def remove_non_ASCII(sentence):

    return ''.join(i for i in sentence if ord(i) < 128)


def tokenize1(sentence):
    #print sentence
    sentence1 = word_tokenize(sentence)
    #print sentence1
    sentence = ' '.join(word1 for word1 in sentence1)
    return sentence

for y, row in trainframe.iterrows():

    row['text'] = remove_non_ASCII(row['text'])
    row['text'] = tokenize1(row['text'])
    row['text'] = removepunct(row['text'])
    row['text'] = row['text'].lower()
    row['text'] = remove_stopwords(row['text'])
    row['text'] = wordsless2(row['text'])
    row['text'] = lemmatizewords(row['text'])

print("removed punctuations,tolower(), stop words, Lemmatize and non-ASCII from train data")
# remove punctuations in test data
for x, row in testframe.iterrows():
    row['text'] = remove_non_ASCII(row['text'])
    row['text'] = tokenize1(row['text'])
    row['text'] = removepunct(row['text'])
    row['text'] = row['text'].lower()
    row['text'] = remove_stopwords(row['text'])
    row['text'] = wordsless2(row['text'])
    row['text'] = lemmatizewords(row['text'])
print("removed punctuations,tolower(), stop words,lemmatize and non-ASCII from test data")


unique_words = []


for a, row in trainframe.iterrows():

    wordBag = word_tokenize(row['text'])

    for word in wordBag:
       unique_words.append(word)



finalist = set(unique_words)
'''
print("Created UNIQUE WORD LIST")
with open('bagofwords1', 'wb') as fp:
    pickle.dump(finalist, fp)
'''
# #Using a count vectorizer
vectorizer = CountVectorizer(vocabulary=finalist,binary=True)
#vectorizer = CountVectorizer(ngram_range=(1, 2))
matrix = vectorizer.fit_transform(trainframe['text'])
#print(vectorizer.get_feature_names())
#print(finalist)
print(matrix.todense())
print(matrix.shape)
print(type(matrix))
#vectorizer2 = CountVectorizer(ngram_range=(1,2) binary=True)
matrix2 = vectorizer.transform(testframe['text'])
print(matrix2.todense())
print(matrix2.shape)
# #USING TEXT FILE METHOD TO CREATE A MATRIX
'''
with open("/Users/harika/Downloads/trainembedding.txt","w") as file1:
    for i, row in trainframe.iterrows():
        for word in finalist:
            if word in row["text"]:
                file1.write("1 ")
            else:
                file1.write("0 ")
        file1.write("\n")
file1.close()
print("completed training data")
with open("/Users/harika/Downloads/testembedding.txt", "w") as file2:
    for i, row in testframe.iterrows():
        for word in finalist:
            if word in row["text"]:
                file2.write("1 ")
            else:
                file2.write("0 ")
        file2.write("\n")
file2.close()
print('completed test data')
mat=[]

trainembedd = np.loadtxt("/Users/harika/Downloads/trainembedding.txt", dtype='int')
testembedd = np.loadtxt("/Users/harika/Downloads/testembedding.txt", dtype='int')

#print(mat)
#print(type(trainembedd))
'''
# #changes the text labels to binary
for counter, row in trainframe.iterrows():
    if row['label'] == "positive":
        row['label'] = 1
    else:
        row['label'] = 0

for counter, row in testframe.iterrows():
    if row['label'] == "positive":
        row['label'] = 1
    else:
        row['label'] = 0

print(trainframe['label'])
trainframe['label'] = trainframe['label'].astype('int')
testframe['label'] = testframe['label'].astype('int')
print(testframe['label'])
with open('actualpredictionsRF', 'wb') as fp:
    pickle.dump(testframe['label'], fp)
print("completed encoding labels")
# #clf = RandomForestClassifier()
#clf = tree.DecisionTreeClassifier(criterion='entropy')
clf=RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf = clf.fit(matrix, trainframe['label'])
#clf = clf.fit(matrix, trainframe['label'])
#predictions = clf.predict(matrix2)
#pickle to save the data
print(clf.decision_path(matrix))
filename = 'randomforestFinal'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict(matrix2)
'''
filename1='RandomForestpredict'
pickle.dump(predictions,open(filename1, 'wb'))
'''
#print(clf.feature_importances_)
#predictions = clf.predict(matrix2)
ts2 = time.time()
ts2 = ts2-ts
print("time taken")
print(ts2)
print("accuracy")
false_positive_rate, true_positive_rate, thresholds = roc_curve(testframe['label'], predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(metrics.accuracy_score(testframe['label'], predictions))
print("completed prediction")
print(confusion_matrix(testframe['label'], predictions))
print(classification_report(testframe['label'], predictions))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()