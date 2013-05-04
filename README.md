Homework_08
===========

Columbia applied data science homework 08

Notice: Please change the data address every time before you run any file! 

To clean the file, first run src/onlyclosed.py to separate the 3.7G train.csv into two files,
one is 61M, the other almost 3.7G may take around 10 mins

Then run scripts/subsample_train.sh to get cutted_equal_closed_open.csv, which is 130m and takes around 5 mins

Then run src/Pickle.py, which will output a token file of 390m costing 10mins

Token file is a pandas.dataframe file that tokenize the text in Title and BodyMarkdown

Three classifiers are logistic.py, NBlogistic.py, and NaiveBayes.py, they all take the token file as input

Because of speed of editing. These classifier by default take only 2% of token file as train data and test data,

so every classifier will result in 2min

