import re
from nltk.corpus import stopwords
import pandas as pd
import time
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV

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

def feature_extraction():


    # Split data into test and train data
    dataframe1 = pd.read_csv('/Users/harika/Downloads/reviews.csv', sep='|', nrows=100000)

    print "Split into training and test data"

    for index, row in dataframe1.iterrows():
        row['text'] = remove_non_ASCII(row['text'])
        row['text'] = tokenize1(row['text'])
        row['text'] = removepunct(row['text'])
        row['text'] = row['text'].lower()
        row['text'] = remove_stopwords(row['text'])
        row['text'] = wordsless2(row['text'])
        row['text'] = lemmatizewords(row['text'])

    unique_words = []
    for a, row in dataframe1.iterrows():

        wordBag = word_tokenize(row['text'])
        for word in wordBag:
            unique_words.append(word)

    finalist = set(unique_words)

    cv = CountVectorizer(vocabulary=finalist)

    embedding_train = cv.fit_transform(dataframe1['text'])

    print "Embedding for train created"

    dataframe1['label'] = pd.Categorical(dataframe1['label'])
    dataframe1['label'] = dataframe1['label'].cat.codes

    print "Labels encoded"



    # Create regularization hyperparameter space
    #C = np.logspace(-2, 4, 10)
    #solver=['adam']
    #learning_rate=[0.01, 0.001, 0.0001]
    #activation = ['tanh','logistic', 'relu', 'identity']
    #max_iterations = [500, 1000, 1500]
    #hidden_layer_sizes=[(10, 10), (10, 10, 10), (20, 20, 20)]

    # Create hyperparameter options
    C = [1,10,100,1000]
    gamma = [0.001,0.0001]
    kernel=['rbf', 'linear']

    #hyperparameters = dict(solver=solver,learning_rate_init=learning_rate,activation=activation,hidden_layer_sizes=hidden_layer_sizes,max_iter=max_iterations)
    hyperparameters = dict(C=C, gamma=gamma, kernel=kernel)
    neuralnet = svm.SVC()

    clf = GridSearchCV(neuralnet, hyperparameters, cv=5, verbose=0)

    X_train, X_test, y_train, y_test = train_test_split(
        embedding_train[0:len(dataframe1)],
        dataframe1['label'],
        train_size=0.80,
        random_state=1234)

    best_model = clf.fit(X_train, y_train)
    # View best hyperparameters
    #print('Best iterations:', best_model.best_estimator_.get_params()['max_iter'])
    #print('Best activation:', best_model.best_estimator_.get_params()['activation'])
    #print('Best hidden layers:', best_model.best_estimator_.get_params()['hidden_layer_sizes'])
    print('Best gamma:', best_model.best_estimator_.get_params()['gamma'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])


# START OF CODE
start_time = time.time()
print("--- %s seconds ---" % (start_time))
feature_extraction()

print("--- %s seconds ---" % (time.time() - start_time))