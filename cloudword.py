from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import time
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pickle
from sklearn.feature_extraction.text import CountVectorizer


ts = time.time()
dataframe1 = pd.read_csv('/Users/harika/Downloads/reviews.csv', sep='|',nrows=100000)
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
for y, row in testframe.iterrows():
    row['text'] = remove_non_ASCII(row['text'])
    row['text'] = tokenize1(row['text'])
    row['text'] = removepunct(row['text'])
    row['text'] = row['text'].lower()
    row['text'] = remove_stopwords(row['text'])
    row['text'] = wordsless2(row['text'])
    row['text'] = lemmatizewords(row['text'])
    #row["text"] = stemming(row['text'])

print("removed punctuations,tolower(), stop words, Lemmatize and non-ASCII from train data")
posreviews=[]
negreviews=[]
for a, row in testframe.iterrows():
    if row['label'] == 'positive':
        posreviews.append(row['text'])
    else:
        negreviews.append(row['text'])

print(len(posreviews))
positivewordBag=[]
negativewordBag=[]
for i in range(len(posreviews)):
    for word in posreviews[i].split():
        positivewordBag.append(word)

for j in range(len(negreviews)):
    for word in negreviews[j].split():
        negativewordBag.append(word)

positivetag=nltk.pos_tag(positivewordBag)
negativetag=nltk.pos_tag(negativewordBag)
verblist=[]
verblist1=[]
dict1={}

#print positivetag
print("created positive and negative wordbag")
for word, pos in positivetag:
    if pos == 'VB' or pos == 'JJ' or pos == 'RB' or pos == 'RBR' or pos == 'RBS':
        verblist.append(word)
for word, pos in negativetag:
    if pos == 'VB' or pos == 'JJ' or pos == 'RB' or pos == 'RBR' or pos == 'RBS':
        verblist1.append(word)

#print(verblist1)
col_count = Counter(verblist)
dict1=col_count
finallist=[]
neglsitCount = Counter(verblist1)
ignore=['get','even','also','many','really','much','first','read','write','new','else','high','give','still']
for word in list(col_count):
    if word in ignore:
        del col_count[word]
negativebag= neglsitCount.most_common(50)
print neglsitCount

print type(col_count)
#print col_count
wordslist = ' '

print("Created list")
#wordcloud = WordCloud(width=1000, height=1000,
#                      background_color='white',
#                      ).generate_from_frequencies(col_count)
# plot the WordCloud image
wordcloud1= WordCloud(width=1200, height=1000,
                      background_color='white', max_words=400
                      ).generate_from_frequencies(col_count)
plt.figure(figsize=(10, 10), facecolor=None)
#plt.imshow(wordcloud)
plt.imshow(wordcloud1)
plt.axis("off")
#plt.tight_layout(pad=0)
plt.show()


