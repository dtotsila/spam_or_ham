
import torch
import pandas as pd
from torch.utils.data import Dataset
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as sw
import re

# text processing
ps = PorterStemmer()
stop_words = set(sw.words("english"))

# check if pandas series text field has a number
def has_numbers(row):
    text = row["text"]
    string = word_tokenize(text)
    for word in string:
        if bool(re.search(r'\d', word)):
            return 1
    return 0

# check if pandas series text field has a currency symbol
def has_currency(row):
    text = row["text"]
    string = word_tokenize(text)
    for word in string:
        if('$' in word or '£' in word or '€' in word):    
            return 1
    return 0

# check if a string has numbers
def has_numbers_text(text):
    string = word_tokenize(text)
    for word in string:
        if bool(re.search(r'\d', word)):
            return 1
    return 0

# check if a string has currency symbols
def has_currency_text(text):
    string = word_tokenize(text)
    for word in string:
        if('$' in word or '£' in word or '€' in word):    
            return 1
    return 0

# calculate text field length from pandas series
def sms_len(row):
    return len(row["text"])

# stem words that are not stop words and are alphabetic strings from text field of pandas series
def processing(row):
    text = row["text"]
    tokens = word_tokenize(text)
    stemmed_tokens = []
    stemmed_tokens = [ps.stem(word.lower()) for word in tokens if (word not in stop_words and word.isalpha())]
    joined = (" ".join(stemmed_tokens))

    return joined

# stem words that are not stop words and are alphabetic strings from string
def message_processing(text):
    tokens = word_tokenize(text)
    stemmed_tokens = []
    stemmed_tokens = [ps.stem(word.lower()) for word in tokens if (word not in stop_words and word.isalpha())]
    joined = (" ".join(stemmed_tokens))
    return word_tokenize(joined)

# list or numpy dataset to torch tensor
class CustomDataset(Dataset):
    """ Custom Spam or Ham Dataset. """

    def __init__(self, x, y):
        """
        Args:
            x (numpy array): # tf-idf vector representation of message
            y (numpy array): # class spam or ham
        """
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# read train or test dataset
def read_data(id):
    '''
        id = test or train
    '''
    # Read Train Data
    df = pd.read_csv('data/' + id +'.csv', index_col=0)
    df.category = pd.factorize(df.category)[0]
    X, y = df.iloc[:, 1:].to_numpy(), df["category"].to_numpy()
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    return X, y

# plot confusion matrix
def plot_cf(conf_matrix,name,show=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix '+name, fontsize=18)
    if show:
        plt.show()
    else:
        plt.savefig("plots/"+name+'_cf.png')

# decorate axis from plot
def decorate_axis(ax, remove_left=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)