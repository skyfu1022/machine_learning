import numpy as np
import re
import os
import random
import feedparser
from bayes import NavieBayes

def text_parse(big_string):
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

# def calculate_most_freq(vocabulary_list, full_text):
#     unique_word, num_each_word = np.unique(full_text, return_counts=True)
#     index = np.concatenate(list(map(lambda x: np.where(vocabulary_list == x)[0], unique_word)))


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    categories = os.listdir('email')
    for i, category in enumerate(categories):
        path = os.path.join('email', category)
        list_dir = os.listdir(path)
        file_list = [os.path.join(path, file) for file in list_dir]
        # ham_list = map(lambda file: os.path.join('email/ham', file), os.listdir('email/ham'))
        # text_parse(open('email/ham/23.txt').read())
        doc_list += [text_parse(open(file).read()) for file in file_list]
        class_list += [i for _ in range(len(file_list))]
    for words in doc_list:
        full_text.append(words)

    rdn_num = np.arange(0, len(doc_list))
    test_set_index = random.sample(list(rdn_num), 10)
    # test_set = np.array(test_set)
    training_set_index = np.delete(rdn_num, test_set_index)
    doc_list = np.array(doc_list)
    class_list = np.array(class_list)
    model = NavieBayes('email_model.txt')
    model.train(doc_list[training_set_index], class_list[training_set_index])
    pred = model.classify(doc_list[test_set_index])
    error = np.sum(pred != class_list[test_set_index]) / 10.0
    print('the error rate is: ', error)
        # doc_list += list(map(lambda file: text_parse(open(file).read()), file_list))
        # doc_list.append(list(map(lambda file: text_parse(open(file).read()), ham_list)))
    # # word_list = spam_list + ham_list
    # labels = np.ones(spam_word_list)
    # labels.append(np.zeros(ham_word_list))
    # return doc_list

def local_words(feeds):
    doc_list = []
    class_list = []
    full_text = []
    # categories = os.listdir('email')
    for label, feed in enumerate(feeds):
        # path = os.path.join('email', category)
        # list_dir = os.listdir(path)
        # file_list = [os.path.join(path, file) for file in list_dir]
        # ham_list = map(lambda file: os.path.join('email/ham', file), os.listdir('email/ham'))
        # text_parse(open('email/ham/23.txt').read())
        num_data = len(feed['entries'])
        doc_list += [text_parse(feed['entries'][i]['summary']) for i in range(num_data)]
        class_list += [label for _ in range(num_data)]
    for words in doc_list:
        full_text.append(words)

    rdn_num = np.arange(0, len(doc_list))
    test_set_index = random.sample(list(rdn_num), 10)
    # test_set = np.array(test_set)
    training_set_index = np.delete(rdn_num, test_set_index)
    doc_list = np.array(doc_list)
    class_list = np.array(class_list)
    model = NavieBayes('email_model.txt')
    model.train(doc_list[training_set_index], class_list[training_set_index])
    pred = model.classify(doc_list[test_set_index])
    error = np.sum(pred != class_list[test_set_index]) / 10.0
    print('the error rate is: ', error)

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craiglist.org/stp/index.rss')
local_words([ny, sf])
# spam_test()
# my_sent = 'this book is the best book on Python or M.L. I have ever laid eyes upon.'
# list_of_tokens = reg_ex.split(my_sent)
# list_of_tokens =
# print(list_of_tokens)
# def load_data_set():
#     posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
#                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
#                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
#                     ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
#                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
#                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
#     class_vec = np.array([0, 1, 0, 1, 0, 1])
#     return posting_list, class_vec
#
# list_of_posts, list_class = load_data_set()
# model = NavieBayes('bayes.txt')
# model.train(list_of_posts, list_class)
# test_entry = ['love', 'my', 'dalmation']
# print(test_entry, 'classitied as: ', model.classify(test_entry))
# test_entry = ['stupid', 'garbage']
# print(test_entry, 'classitied as: ', model.classify(test_entry))
