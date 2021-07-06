import pandas as pd
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
input_set = pd.read_csv("Dataset.csv")
# print(input_set.head())
input_set.drop('Prep_obj', axis=1)
#Label Encoding
label_encoding = preprocessing.LabelEncoder()
input_set['verb_n'] = label_encoding.fit_transform(input_set['Verb'])
input_set['noun_n']  = label_encoding.fit_transform(input_set['Noun'])
input_set['class_n'] = label_encoding.fit_transform(input_set['Class_label'])
input_set['prep_n'] = label_encoding.fit_transform(input_set['Prep'])

# print(input_set.head())
#getting input and output colmun
inputs_n = input_set.drop(['Verb', 'Noun','Class_label', 'Prep','Prep_obj', 'class_n'], axis=1)
target = input_set['class_n']
print(inputs_n.head())
print(target.head())

X_train, X_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2, random_state=1)
#print(X_test)

#useing naviebais

naive = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
naive.fit(X_train,y_train)
pred = naive.predict(X_test)
print(np.mean(pred == y_test))

