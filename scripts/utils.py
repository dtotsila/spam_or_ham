from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as sw
import re

def has_numbers(row):
    text = row["text"]
    string = word_tokenize(text)
    for word in string:
        if bool(re.search(r'\d', word)):
            return 1
    return 0

def has_currency(row):
    text = row["text"]
    string = word_tokenize(text)
    for word in string:
        if('$' in word or '£' in word or '€' in word):    
            return 1
    return 0

def sms_len(row):
    return len(row["text"])
# text processing
ps = PorterStemmer()
stop_words = set(sw.words("english"))

# stem words that are not stop words and are alphabetic strings
def processing(row):
    text = row["text"]
    length = len(text)
    tokens = word_tokenize(text)
    stemmed_tokens = []
    has_num = 0
    has_money = 0
    stemmed_tokens = [ps.stem(word.lower()) for word in tokens if (word not in stop_words and word.isalpha())]
    joined = (" ".join(stemmed_tokens))

    return joined
