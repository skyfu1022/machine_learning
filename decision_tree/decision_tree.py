import numpy as np
import pickle


class ID3:

    def __init__(self, data_set=None, feat_names=None):
        self.data_set = data_set
        self.feat_names = feat_names
        self.tree = {}

    def shannon_entropy(self, data_set):
        """ 
        calulate shannon entropy, entropy = -∑p(x) * log p(x) 
        :param data_set: input array, data matrix，the last column is the class label 
        :return: float, shannon entropy 
        """
        # number of data
        num_entries = len(data_set)

        # get the number of each class
        num_labels, num_each_label = np.unique(data_set[:, -1], return_counts=True)

        # caculate probabilities of each class
        probs = num_each_label / num_entries

        # calucate shannon entropy
        entropy = np.sum(-probs * np.log2(probs))
        return entropy

    def split_data_set(self, data_set, col):
        """ 
        split the data set by the specified col and return the split sub data sets
        :param data_set: numpy array like, the input data set 
        :param col: int, the specified column
        :return: dict, the split sub data sets, the keys of the dict are the unique values appear
                in the specified column feature vector
        """
        # get the vector of the specified col
        feat = data_set[:, col]
        data_set = np.delete(data_set, [col], axis=-1)

        # get the unique classes dict and the number of each unique classes
        feat_values, num_of_each_feat = np.unique(feat, return_counts=True)

        # split the data set by each unique class
        sub_data_sets = {}
        for feat_value in feat_values:
            value_idx = np.where(feat == str(feat_value))[0]
            sub_data_sets[feat_value] = data_set[value_idx, :]
        return sub_data_sets

    def select_max_gain_feat(self, data_set):
        """ 
        select the feature with max information gain  
        :param data_set: numpy array_like, the input data set
        :return best_feature: int, the index of the best feature
        """
        # get the number of features
        num_features = len(data_set[0]) - 1
        base_entropy = self.shannon_entropy(data_set)
        best_info_gain = 0.0
        best_feature = -1

        # caculate information gain of each feature and pick out the max information gain
        for col in range(num_features):

            # caculate the conditional entropy of the specified feature
            sub_data_sets = self.split_data_set(data_set, col)
            new_entropy = 0.0
            for sub_data_set in sub_data_sets.values():
                prob = len(sub_data_set) / len(data_set)
                new_entropy += prob * self.shannon_entropy(sub_data_set)

            # caculate the information gain of the specified feature
            info_gain = base_entropy - new_entropy

            # pick out the feature index of max information gain
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = col
        return best_feature

    def save(self, file_name):
        fw = open(file_name, 'wb')
        pickle.dump(self.tree, fw, 0)
        fw.close()

    def load(self, file_name):
        fr = open(file_name, 'rb')
        return pickle.load(fr)

    def create_tree(self, data_set, feats):
        """
        create the decision tree
        :param data_set: numpy array like, the input data set
        :param feats: list, the list of the feature name
        :return: dict, the desired decision tree
        """
        class_list = data_set[:, -1]

        # if all the class label are the same then stop split
        class_name, num_each_class = np.unique(class_list, return_counts=True)
        if len(class_name) == 1:
            return class_list[0]

        # when traverse all the feature then return the class label class with the most occurrence
        if len(data_set[0]) == 1:
            return class_name[np.argmax(num_each_class)]

        # select the max gain feature
        best_feat = self.select_max_gain_feat(data_set)

        # get the best feature name
        best_feat_label = feats[best_feat]

        # take the best feature name as root of the tree
        my_tree = {best_feat_label: {}}
        # create the decision recursively
        del (feats[best_feat])
        sub_data_sets = self.split_data_set(data_set, best_feat)
        for value in sub_data_sets.keys():
            sub_labels = feats[:]
            my_tree[best_feat_label][value] = self.create_tree(sub_data_sets[value], sub_labels)
        return my_tree

    def train(self):
        data_set = self.data_set.copy()
        feat_names = self.feat_names[:]
        self.tree = self.create_tree(data_set, feat_names)
        return self.tree

    def classify(self, input_tree, test_vec):
        """
        classify the test vector through the trained decision tree
        :param input_tree: dict, the trianed decision tree model
        :param test_vec: numpy array like, the test sample
        :return: str, the classify result
        """
        # get the root feature name
        first_str = list(input_tree.keys())[0]

        # get the child of the root feature
        second_dict = input_tree[first_str]

        # find out the index of the root feature in the label vector
        feat_index = self.feat_names.index(first_str)

        # recurit to find right class
        for key in second_dict.keys():
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.classify(second_dict[key], test_vec)
                else:
                    class_label = second_dict[key]
        return class_label