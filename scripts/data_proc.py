#!/usr/bin/env python
# coding: utf-8
from sklearn.model_selection import train_test_split
from time import time
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NLP stuff
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import processing, has_numbers, has_currency, sms_len

# Read dataset into Pandas-Data Frame  beware of the encoding
df = pd.read_csv('data/dataset.csv', encoding="ISO-8859-1")

# since columns Unamed: # are empty we need to drop them and keep only the usefull data (v1, v2) tag and sms respectively
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Replace the column names so that they are more identifiable and easily accesed
df = df.rename(columns={"v1": "category", "v2": "text"})

# apply processing to every SMS and store results to a new row
df['processed'] = df.apply(processing, axis=1)

# add a new rowsthat indicate if the message has numerical values
df['has_num'] = df.apply(has_numbers, axis=1)

# add a new rowsthat indicate if the message has currency related stuff
df['has_money'] = df.apply(has_currency, axis=1)
# add a new rows that contain the length of the message

df['length'] = df.apply(sms_len, axis=1)
df = df.drop(['text'], axis=1)

# storing data frame to csv so that it can be use from future py script
df.to_pickle('data/stemmed.pkl')


# tfidf vector representation of texts
Transformer = TfidfVectorizer(max_features=2500)
tfidf = Transformer.fit_transform(df.processed.values.astype('U'))
tfidfDF = pd.DataFrame(tfidf.todense())
tfidfDF.columns = sorted(Transformer.vocabulary_)
concaten = pd.concat([df, tfidfDF], axis=1)

concaten = concaten.drop(['processed'], axis=1)
print(concaten)
concaten.to_pickle(r'data/tfidf.pkl')


# X_train, X_test, Y_train, Y_test = train_test_split(
#     concaten.iloc[:, 2:], df['category'], test_size=0.2)

# t0 = time()
# model = GaussianNB()
# model.fit(X_train, Y_train)
# print(f"\nTraining time: {round(time()-t0, 3)}s")
# t0 = time()
# score_train = model.score(X_train, Y_train)
# # print(f"Prediction time (train): {round(time()-t0, 3)}s")t0 = time()
# score_test = model.score(X_test, Y_test)
# # print(f"Prediction time (test): {round(time()-t0, 3)}s")print(“\nTrain set score:", score_train)
# print("Test set score:", score_test)
