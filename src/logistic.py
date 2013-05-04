import cPickle as pickle
import csv
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.linear_model import LogisticRegression
import random
import nltk
from homework_08.src.tagger import document_features, significantWords



data_file = '/media/software/data/processed/token'


with open(data_file, 'r') as f:
    df = pickle.load(f)



df.ix[df.Status != 'open', 'Status'] = 'closed'



rows = random.sample(df.index, len(df)/100)
train_data = df.ix[rows]
test_data = df.ix[random.sample(df.drop(rows).index, len(df)/100)]
#test_data = df.drop(rows)



train_body_status = [tuple(x) for x in train_data[['Body','Status']].values]



significant_words = significantWords(train_body_status)
high_info_words = significant_words['total']
high_open_words = significant_words['open']
high_closed_words = significant_words['closed']
print len(high_info_words)



def getFeatures(question):
    feature = question[['Reputation','Answer']]
    for word in high_info_words:
        feature[word] = 0
    for i in question.index:
        for word in high_info_words:
            if word in question['Body'][i]:
                feature[word][i] = 1
    return feature
                



train_data.Status = train_data.Status.apply(lambda x: 0 if x == 'open' else 1)
test_data.Status = test_data.Status.apply(lambda x: 0 if x == 'open' else 1)



X_train = getFeatures(train_data)
y_train = pd.DataFrame(train_data['Status'])
X_test = getFeatures(test_data)
y_test = pd.DataFrame(test_data['Status'])



classifier = LogisticRegression()
classifier.fit(X_train,y_train)
#classifier.predict_proba(X_test)
print classifier.score(X_test,y_test)
#classifier.coef_




