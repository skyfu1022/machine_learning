import numpy as np
import os
import pickle

class NavieBayes:

    def __init__(self, filename):
        self.filename = filename
        self.model = None

    def create_vocabulary_list(self, data_set):
        """
        create vocabulary list that appear in the data_set
        :param data_set: list[list], the input data set
        :return: list, the desired vocabulary list
        """
        vocabularies_set = set([])
        for data in data_set:
            vocabularies_set = vocabularies_set | set(data)
        return np.array(list(vocabularies_set))

    def set_of_words_to_vector(self, data_set, vocabulary_list):
        """
        convert words to vector
        :param data_set: str list, the input data
        :param vocabulary_list: str list, vocabulary list of the data set
        :return: return_vec, int list, vectors that represent which word appear in the input_data
        """
        train_matrix = []

        # transform each data in data set to vector
        for data in data_set:

            # initialize data vector
            data_vector = np.zeros(len(vocabulary_list))
            data = np.array(data)

            # find whether elements of vocabulary_list are in data
            index1 = np.isin(vocabulary_list, data)

            # data_vector = map(lambda x: data_vector[index[x]] = num_each_word[x], range(len(index)))
            # if words of vocabulary_list appear in data, then set values of data vector in the corresponding location to 1
            data_vector[index1] = 1

            # if words in data do not appear in vocabulary list, then find them and print out
            index2 = np.logical_not(np.isin(data, vocabulary_list))
            if True is index2:
                print('the words: %s are not in my Vocabulary!' % data[index2])
            train_matrix.append(data_vector)
        return np.array(train_matrix)

    def bag_of_words_to_vector(self, data_set, vocabulary_list):
        """
        convert words to vector
        :param data_set: str list, the input data
        :param vocabulary_list: str list, vocabulary list of the data set
        :return: return_vec, int list, vectors that represent which word appear in the input_data
        """
        train_matrix = []

        # transform each data in data set to vector
        for data in data_set:

            # initialize data vector
            data_vector = np.zeros(len(vocabulary_list))

            # get the unique categories and its counts
            unique_data_words, num_each_word = np.unique(data, return_counts=True)

            # find the index of the elements appear in the data in the vocabulary list
            index = np.concatenate(list(map(lambda x: np.where(vocabulary_list == x)[0], unique_data_words)))

            # if words of vocabulary_list appear in data, then set values of data vector in the corresponding location to the counts of words
            data_vector[index] = num_each_word

            # if words in data do not appear in vocabulary list, then find them and print out
            index2 = np.logical_not(np.isin(data, vocabulary_list))
            if True is index2:
                print('the words: %s are not in my Vocabulary!' % data[index2])
            train_matrix.append(data_vector)
        return np.array(train_matrix)

    def train_naive_bayes(self, data_set, labels):
        """
        train naive bayes, calculate the probability of each class 
        :param data_set: 
        :param labels: 
        :return:  
        """
        # create vocabulary list of the data set
        vocabulary_list = self.create_vocabulary_list(data_set)

        # transform the original data set to binary data matrix according the vocabulary list
        trian_data_matrix = self.set_of_words_to_vector(data_set, vocabulary_list)
        num_train_data = len(trian_data_matrix)
        # num_words = len(trian_data_matrix[0])
        # p_abusive = np.sum(labels) / np.float(num_train_docs)

        # find unique categories and its corresponding counts
        unique_class, num_each_class = np.unique(labels, return_counts=True)
        num_classes = len(unique_class)

        # calculate probability of each category
        class_probs = num_each_class / float(num_train_data)
        # num_per_word_class = np.zeros((num_classes, num_words), dtype=np.float)
        # prob_denom = np.ones(num_classes) + 1.0
        cond_probs = []
        for i in range(num_classes):
            # num_per_word_class[i, :] = np.sum(trian_data_matrix[np.where(labels == i)[0], :], axis=0)
            # prob_denom = np.sum(num_per_word_class[i, :], axis=-1) + 2
            # cond_prob_vecs.append(np.log((num_per_word_class[i, :] + 1.0) / prob_denom))

            # calculate the amounts of each words that appear in the specified category i
            num_each_word = np.sum(trian_data_matrix[np.where(labels == i)[0], :], axis=0)

            # calculate the total amount of the words appear in the specified category i
            num_total_words = np.sum(num_each_word, axis=-1)

            # calculate conditional probability of each word given the specified category i
            cond_probs.append(np.log((num_each_word + 1.0) / (num_total_words + 2.0)))
        cond_probs = np.array(cond_probs)
        return {'class_probs': class_probs, 'cond_probs': cond_probs, 'vocabularies': vocabulary_list}

    def train(self, data_set, labels):
        self.model = self.train_naive_bayes(data_set, labels)
        self.save()

    def classify(self, input_data):
        if self.model is None:
            if os.path.exists(self.filename):
                self.load()
            else:
                raise ValueError('model is not exist, please train model!')
        if np.ndim(input_data) == 1:
            np.expand_dims(input_data, axis=0)
        data_matrix = self.set_of_words_to_vector(input_data, self.model['vocabularies'])
        a = np.repeat(np.expand_dims(data_matrix, axis=1), 2, 1)
        prob = np.sum(a * self.model['cond_probs'], axis=-1) + np.log(self.model['class_probs'])
        return np.argmax(prob, axis=-1)

    def save(self):
        fw = open(self.filename, 'wb')
        pickle.dump(self.model, fw, 0)
        fw.close()

    def load(self):
        fr = open(self.filename, 'rb')
        self.model = pickle.load(fr)