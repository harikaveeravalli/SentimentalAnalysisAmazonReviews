import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import time
from sklearn import metrics
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

ts = time.time()
dataframe1 = pd.read_csv('/Users/harika/Downloads/reviews.csv', sep='|')
#dataframe1 = pd.read_csv('/Users/harika/PycharmProjects/sentimentalanalysis/reviews.csv', sep='|')
testframe = dataframe1.iloc[4::5, :].copy()
trainframe = dataframe1.drop(testframe.index)
trainframe.index = range(len(trainframe))
testframe.index = range(len(testframe))

def wordsless2(Sentence):

    Sentence = re.sub(r'\W*\b\w{1,2}\b', '', Sentence)
    return Sentence


def removepunct(sentence):

    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    return sentence

stop_words = set(stopwords.words('english'))


def remove_stopwords(Sentence):
    Sentence = ' '.join(word for word in Sentence.split() if word not in stop_words)
    return Sentence


def lemmatizewords(Sentence):
    lemmatizer = WordNetLemmatizer()

    Sentence = ' '.join(str(lemmatizer.lemmatize(word1, pos='v')) for word1 in Sentence.split())
    return Sentence


def remove_non_ASCII(sentence):

    return "".join(i for i in sentence if ord(i) < 128)

def tokenize1(sentence):

    sentence1 = word_tokenize(sentence)
    sentence = ' '.join(word1 for word1 in sentence1)
    return sentence
'''
def stemming(sentence):
    sentence = ' '.join(porter_stemmer.stem(word) for word in sentence.split())
    return sentence
'''
for y, row in trainframe.iterrows():
    row['text'] = remove_non_ASCII(row['text'])
    row['text'] = tokenize1(row['text'])
    row['text'] = removepunct(row['text'])
    row['text'] = row['text'].lower()
    row['text'] = remove_stopwords(row['text'])
    row['text'] = wordsless2(row['text'])
    row['text'] = lemmatizewords(row['text'])
    #row["text"] = stemming(row['text'])

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

'''
unique_words = []
for a, row in trainframe.iterrows():

    wordBag = word_tokenize(row['text'])
    for word in wordBag:
        unique_words.append(word)



finalist = set(unique_words)
'''
with open('/Users/harika/PycharmProjects/sentimentalanalysis/naiveBagWords','rb') as fp:
    itemlist = pickle.load(fp)

finallist=itemlist
print("VOCABULARY created")

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


vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(trainframe['text'])
print(matrix.todense())
print(matrix.shape)
print(type(matrix))
#vectorizer2 = CountVectorizer(ngram_range=(1,2) binary=True)
matrix2 = vectorizer.transform(testframe['text'])
print(matrix2.todense())
print(matrix2.shape)
classifier = MultinomialNB()
classifier.fit(matrix, trainframe['label'])
filename = 'naivemultibigrampickle'
pickle.dump(classifier, open(filename, 'wb'))
predictions = classifier.predict(matrix2)
ts2 = time.time()
ts2 = ts2-ts
print("done prediction")
print("time taken")
print(ts2)
print("accuracy")
print(metrics.accuracy_score(testframe['label'], predictions))
print(confusion_matrix(testframe['label'], predictions))
print(classification_report(testframe['label'], predictions))

