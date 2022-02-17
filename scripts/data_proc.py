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

# we will balance the dataset a little bit 
# under sample ham (select half th rows)
ham = df[df['category'] == 'ham']
# get 50% of ham demos
ham = ham.sample(frac=0.5)

# over sample spam (select each row two times)
spam = df[df['category'] == 'spam']
# duplicate every demo
spam = pd.concat([spam]*2, ignore_index=True) 

# concatenate spam and ham demos
df = pd.concat([spam,ham], ignore_index=True)
# shuffle dataset
df = df.sample(frac=1)

# apply processing to every SMS and store results to a new column
df['processed'] = df.apply(processing, axis=1)

# add a new column that indicates if the message has numerical values
df['has_num'] = df.apply(has_numbers, axis=1)

# add a new column that indicates if the message has currency related stuff
df['has_money'] = df.apply(has_currency, axis=1)
# add a new column that contains the length of each message

df['length'] = df.apply(sms_len, axis=1)
df = df.drop(['text'], axis=1)



# tfidf vector representation of texts
Transformer = TfidfVectorizer(max_features=2500)
tfidf = Transformer.fit_transform(df.processed.values.astype('U'))
tfidfDF = pd.DataFrame(tfidf.todense())
tfidfDF.columns = sorted(Transformer.vocabulary_)
final = pd.concat([df, tfidfDF], axis=1)
final = final.drop(['processed'], axis=1)
print(final)


# split and store train-test because every method will be trained and tested on the same data, so that the comparisons are as fair as possible 
train, test = train_test_split(final, test_size=0.33)
train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
