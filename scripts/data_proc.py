#!/usr/bin/env python
# coding: utf-8

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NLP stuff
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer


# Read dataset into Pandas-Data Frame  beware of the encoding
df = pd.read_csv('data/dataset.csv', encoding="ISO-8859-1")

# since columns Unamed: # are empty we need to drop them and keep only the usefull data (v1, v2) tag and sms respectively
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Replace the column names so that they are more identifiable and easily accesed
df = df.rename(columns={"v1": "category", "v2": "text"})
print(df.head())

# Some plots showing the dataset distribution
ham_count = len(df[(df["category"] == "ham")])
spam_count = len(df[(df["category"] == "spam")])

# Dataset Distribution
fig, ax = plt.subplots()

# Example data
categories = ['Spam', 'Ham']
count = [spam_count, ham_count]
ax.bar(categories, count)
plt.show()
# We have an inbalanced dataset 86:14 so either undersampling or downsamplin

# text processing

ps = PorterStemmer()
stop_words = set(sw.words("english"))

# stem words that are not stop words and are alphabetic strings


def processing(row):
    text = row["text"]
    tokens = word_tokenize(text)
    stemmed_tokens = [ps.stem(word.lower()) for word in tokens if (
        word.isalpha() and word not in stop_words)]
    joined = (" ".join(stemmed_tokens))
    return joined


# apply processing to every SMS and store results to a new row
df['processed'] = df.apply(processing, axis = 1)
df = df.drop(['text'], axis = 1)
# storing data frame to csv so that it can be use from future py script
df.to_pickle('data/stemmed.pkl')


# tfidf vector representation of texts
Transformer = TfidfVectorizer(max_features = 2000)
tfidf = Transformer.fit_transform(df.processed.values.astype('U'))
tfidfDF = pd.DataFrame(tfidf.todense())
tfidfDF.columns = sorted(Transformer.vocabulary_)
concaten = pd.concat([df, tfidfDF], axis = 1)
concaten = concaten.drop(['processed'], axis = 1)

concaten.to_pickle(r'data/tfidf.pkl')


# split ham and spam
# rows = 5572
ham_df = concaten[concaten['category'] == 'ham']
# rows = 747
spam_df = concaten[concaten['category'] == 'spam'] 

# under sample

ham_under_sampled = ham_df[:1000]

under_sampled_df = pd.concat([ham_under_sampled, spam_df])
under_sampled_df = under_sampled_df.sample(frac=1).reset_index(drop=True)
under_sampled_df.to_pickle(r'data/under_sampled.pkl')


spam_over_sampled = pd.concat([spam_df]*4, ignore_index=True) # Ignores the index
over_sampled_df = pd.concat([spam_over_sampled, ham_df])
over_sampled_df = over_sampled_df.sample(frac=1).reset_index(drop=True)
over_sampled_df.to_pickle(r'data/over_sampled.pkl')

