# initial data import
import pandas
import numpy
import re
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

# data cleaning
"""
1. capture classification output report
2. list out the features extracted 
3. provide an input box for user to input review
4. provide output box for user to see the features extracted
"""

data = pandas.read_csv('Tweets-transformed.csv')