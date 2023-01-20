#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

# NLP stuff
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import processing, has_numbers, has_currency, sms_len, decorate_axis
import matplotlib.pyplot as plt

# set to True if you want to plot stuff
diag = False

# set to True if you want to show plots instead of saving them
show = False


# Read dataset into Pandas-Data Frame  beware of the encoding
df = pd.read_csv('data/dataset.csv', encoding="ISO-8859-1")

# since columns Unamed: # are empty we need to drop them and keep only the usefull data (v1, v2) tag and sms respectively
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Replace the column names so that they are more identifiable and easily accesed
df = df.rename(columns={"v1": "category", "v2": "text"})

# we will balance the dataset a little bit
ham = df[df['category'] == 'ham']
spam = df[df['category'] == 'spam']

# plot dataset distribution before undersampling
if diag == True:
    colors = ['slateblue', 'darkorange']
    plt.pie([len(spam), len(ham)], labels=['spam', 'ham'],
            colors=colors, explode=[0.2, 0],  autopct='%1.1f%%')
    plt.legend()
    plt.title('Dataset Distribution Before')
    if show:
        plt.show()
    else:
        plt.savefig("plots/data_dist_bef.png")
    plt.close()

# get 70% of ham demos
ham = ham.sample(frac=0.7)

# plot dataset distribution after undersampling
if diag == True:
    colors = ['slateblue', 'darkorange']
    plt.pie([len(spam), len(ham)], labels=['spam', 'ham'],
            colors=colors, explode=[0.2, 0],  autopct='%1.1f%%')
    plt.legend()
    plt.title('Dataset Distribution After')
    if show:
        plt.show()
    else:
        plt.savefig("plots/data_dist_aft.png")
    plt.close()

# create and plot a list that contains number of words in  messages and their count of occurence in the dataset
if diag == True:

    dataset_ham_count = ham['text'].str.split().str.len()
    dataset_ham_count.index = dataset_ham_count.index.astype(str) + ' words:'
    dataset_ham_count.sort_index(inplace=True)

    dataset_spam_count = spam['text'].str.split().str.len()
    dataset_spam_count.index = dataset_spam_count.index.astype(str) + ' words:'
    dataset_spam_count.sort_index(inplace=True)
    bins = np.linspace(0, 50, 10)
    fig = plt.figure()
    ax0 = fig.add_subplot()

    plt.title('Number of words in each category and count of occurence distribution')
    ax0.hist([dataset_spam_count, dataset_ham_count],
             bins, label=['spam', 'ham'], color=colors)
    decorate_axis(ax0)
    plt.legend(loc='upper right')
    plt.xlabel("Number of words")
    plt.ylabel("Message count")
    if show:
        plt.show()
    else:
        plt.savefig("plots/words_per_cat.png")
    plt.close()

# concatenate spam and ham demos
df = pd.concat([spam, ham], ignore_index=True)

# shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# apply processing to every SMS and store results to a new column
df['processed'] = df.apply(processing, axis=1)

# add a new column that indicates if the message has numerical values
df['has_num'] = df.apply(has_numbers, axis=1)

# add a new column that indicates if the message has currency related stuff
df['has_money'] = df.apply(has_currency, axis=1)

# add a new column that contains the length of each message
df['length'] = df.apply(sms_len, axis=1).values.reshape(-1, 1)

# the original message is no longer needed
df = df.drop(['text'], axis=1)

# tfidf vector representation of texts
Transformer = TfidfVectorizer(max_features=2500, max_df=0.8)
tfidf = Transformer.fit_transform(df.processed.values.astype('U'))
tfidfDF = pd.DataFrame(tfidf.todense())
tfidfDF.columns = sorted(Transformer.vocabulary_)

# store idf of corpus needed for generatoin of tf idf for new (unseen text)
idf = pd.DataFrame(Transformer.idf_).transpose()
idf = pd.DataFrame(data=idf.values, columns=tfidfDF.columns)

final = pd.concat([df, tfidfDF], axis=1)
final = final.drop(['processed'], axis=1)

# split and store train-test because every method will be trained and tested on the same data, so that the comparisons are as fair as possible
train = final.head(int(np.floor(len(final) * 0.8)))
test = final.tail(int(np.ceil(len(final) - len(final) * 0.8)))

train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
