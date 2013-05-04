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
#test_data = df.drop(rows)
test_data = df.ix[random.sample(df.drop(rows).index, len(df)/100)]


train_body_status = [tuple(x) for x in train_data[['Body','Status']].values]
test_body_status = [tuple(x) for x in test_data[['Body','Status']].values]



significant_words = significantWords(train_body_status)
high_info_words = significant_words['total']
high_open_words = significant_words['open']
high_closed_words = significant_words['closed']



def getNBfeatures(document):
    
    document_words = set(document)#set([token for (token, pos) in document])
    last_words = set(document[-50:])#set([token for (token, pos) in document][-50:])
    n_open_word_in_doc = len(high_open_words & document_words)
    n_closed_word_in_doc = len(high_closed_words & document_words)

    features = {}
    
    for word in high_info_words:
        if word in document_words:
            features['has(%s)' % word] = True

    if len(high_open_words & last_words) > 1:
        features['has open word in last_words'] = True
    if len(high_closed_words & last_words) > 0:
        features['has closed word in last_words'] = True


    if float(n_open_word_in_doc)/len(document)>0.005:#n_open_word_in_doc >7
        features['has many open words'] = True

    if float(n_closed_word_in_doc)/len(document)>0.005:#n_closed_word_in_doc >4
        features['has many closed words'] = True
    

    return features


train_tuples =[(getNBfeatures(token_list), status) for (token_list, status) in train_body_status]
test_tuples =[(getNBfeatures(token_list), status) for (token_list, status) in test_body_status]
classifier = nltk.NaiveBayesClassifier.train(train_tuples)




def getNBposterior(token_list):
    features_status_tuple = getNBfeatures(token_list)
    return classifier.prob_classify(features_status_tuple).prob('closed')



def getLogisticFeature(question):
    feature = question[['Reputation','Answer']]
    feature['BodyNBposterior'] = 0.
    for i in question.index:
        feature['BodyNBposterior'][i] = getNBposterior(question['Body'][i])
    return feature
        

X_train = getLogisticFeature(train_data)
X_test = getLogisticFeature(test_data)

y_train = pd.DataFrame(train_data['Status'].apply(lambda x: 0 if x == 'open' else 1))
y_test = pd.DataFrame(test_data['Status'].apply(lambda x: 0 if x == 'open' else 1))


classifier = LogisticRegression()
classifier.fit(X_train,y_train)
#classifier.predict_proba(X_test)
print 'accuracy rate:', classifier.score(X_test,y_test)
#classifier.coef_


