import cPickle as pickle
import csv
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.linear_model import LogisticRegression
import random
import nltk


data_file = '../data/processed/cutted_equal_closed_open.csv'
out_file = '/media/software/data/processed/token'


df = read_csv(data_file)


df.columns = ['Reputation', 'Answer', 'Title', 'Body', 'Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5', 'Status']


for i in df.index:
    df.ix[i, 'Body'] = nltk.word_tokenize(df.ix[i, 'Body'])
    df.ix[i, 'Title'] = nltk.word_tokenize(df.ix[i, 'Title'])


with open(out_file, 'w') as f:
    pickle.dump(df, f)



