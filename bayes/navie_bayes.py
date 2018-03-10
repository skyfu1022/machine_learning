import numpy as np

def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return return_vec


def train_naive_bayes(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = np.sum(train_category) / np.float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] ==1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify(vec_to_classify, p0_vec, p1_vec, p_class1):
    p1 = np.sum(vec_to_classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec_to_classify * p0_vec) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0

def set_of_words_to_veca(vocab_list, input_data):
        '''
        convert words to vector
        :param vocab_list: str list, vocabulary list that appear in the data_set
        :param input_data: str list, the input data
        :return: return_vec, int list, vectors that represent which word appear in the input_data
        '''
        return_vec = np.zeros(len(vocab_list))
        data = np.array(input_data)
        index1 = np.isin(vocab_list, data)
        return_vec[index1] = 1
        index2 = np.logical_not(np.isin(data, vocab_list))
        if True is index2:
            print('the words: %s are not in my Vocabulary!' % data[index2])
        return return_vec


def testing_nb():
    list_of_posts, list_class = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)
    train_mat = []

    for post_in_doc in list_of_posts:
        train_mat.append(set_of_words_to_vec(my_vocab_list, post_in_doc))
    p0_v, p1_v, p_ab = train_naive_bayes(np.array(train_mat), np.array(list_class))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words_to_vec(my_vocab_list, test_entry))
    print(test_entry, 'classitied as: ', classify(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words_to_vec(my_vocab_list, test_entry))
    print(test_entry, 'classitied as: ', classify(this_doc, p0_v, p1_v, p_ab))


def main():
    testing_nb()

if __name__ == '__main__':
    main()
