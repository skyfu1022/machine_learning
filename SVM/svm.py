import numpy as np
# from sklearn import svm

# svm.SVC()


class SVM(object):
    def __init__(self, data_matrix, label_matrix, c, epsilon, kernel_tuple=('liner', 0)):
        self.data_matrix = data_matrix
        self.label_matrix = label_matrix
        self.c = c
        self.epsilon = epsilon
        self.num_data = data_matrix.shape[0]
        self.alphas = np.matrix(np.zeros((self.num_data, 1)))
        self.kernel_tuple = kernel_tuple
        self.bias = 0
        self.error_caches = np.matrix(np.zeros((self.num_data, 2)))
        self.kernel_trans = np.matrix(np.zeros((self.num_data, self.num_data)))
        for i in range(self.num_data):
            self.kernel_trans[:, i] = self.kernel(self.data_matrix, self.data_matrix[i, :])

    def clip_alpha(self, alpha, l, h):
        if alpha > h:
            alpha = h
        if alpha < l:
            alpha = l
        return alpha

    def select_second_idx_random(self, first_idx):
        """
        random select index of the second parameter 
        :param first_idx: int, the index of the first parameter
        :return: int, the index of the second parameter
        """
        second_idx = first_idx
        while second_idx == first_idx:
            second_idx = np.random.randint(0, self.num_data)
        return second_idx

    def select_second_idx(self, first_idx, error_first_idx):
        """
        select index of the second parameter by the maximum error
        :param first_idx: 
        :param error_first_idx: 
        :return: 
        """
        # initialize the max_error_index, max_delta_error and error_second_idx
        max_error_idx = -1
        max_delta_error = 0
        error_second_idx = 0

        # update the error of first_idx sample to error caches
        self.error_caches[first_idx] = [1, error_first_idx]

        # get the index of sample with non boundary value alpha
        valid_cache_list = np.nonzero(self.error_caches[:, 0].A)[0]

        # if valid_cache_list is not empty, them select the second alpha that need to be
        # optimized through max error sample from the valid_cache_list
        if len(valid_cache_list) > 1:
            for candidate_idx in valid_cache_list:
                if candidate_idx == first_idx:
                    continue
                error_candidate_param = self.calculate_error(candidate_idx)
                delta_error = np.abs(error_first_idx - error_candidate_param)
                if delta_error > max_delta_error:
                    max_error_idx = candidate_idx
                    max_delta_error = delta_error
                    error_second_idx = error_candidate_param
            return max_error_idx, error_second_idx

        # else random select the second variable that need to be optimized
        else:
            j = self.select_second_idx_random(first_idx)
            error_j = self.calculate_error(j)
        return j, error_j

    def calculate_error(self, index):
        """
        calculate error between f(data_matrix[index]) and label_matrix[index]
         E = (alpha * y.T) * (X * x.T) + b
        :param index: int, the index of the data
        :return: float, the error
        """
        f = np.float(np.multiply(self.alphas, self.label_matrix).T * self.kernel_trans[:, index]) + self.bias
        error = f - np.float(self.label_matrix[index])
        return error

    def update_error_index(self, index):
        error_index = self.calculate_error(index)
        self.error_caches[index] = [1, error_index]

    def kernel(self, x, y):
        # x = self.data_matrix[first_idx, :]
        # y = self.data_matrix[second_idx, :]
        m, n = np.shape(x)
        kernel_trans = np.matrix(np.zeros((m, 1)))
        if self.kernel_tuple[0] == 'liner':
            k = x * y.T
        elif self.kernel_tuple[0] == 'rbf':
            for j in range(m):
                delta = x[j, :] - y
                kernel_trans[j] = delta * delta.T
            return np.exp(kernel_trans / (- 2 * self.kernel_tuple[1] ** 2))
        else:
            raise NameError('Houston We Have a Problem -- hat Kernel is not recognized')

    def inner_loop(self, first_idx):
        """
        the inner loop of patt-smo algorithm
        :param first_idx: int, index of the first selected alpha
        :return: 
        """
        # calculate the error of the first sample
        error_first_idx = self.calculate_error(first_idx)

        # judge whether the KKT condition is violated
        if ((self.label_matrix[first_idx] * error_first_idx < -self.epsilon) and (
                self.alphas[first_idx] < self.c)) or (
                (self.label_matrix[first_idx] * error_first_idx > self.epsilon) and self.alphas[first_idx] > 0):

            # select the first alpha that need to be optimized
            second_idx, error_second_idx = self.select_second_idx(
                first_idx, error_first_idx)
            alpha_i_old = self.alphas[first_idx].copy()
            alpha_j_old = self.alphas[second_idx].copy()

            # calculate the boundary value of the two alpha
            if self.label_matrix[first_idx] != self.label_matrix[second_idx]:
                l = max(0, self.alphas[second_idx] -
                        self.alphas[first_idx])
                h = min(self.c, self.c +
                        self.alphas[second_idx] - self.alphas[first_idx])
            else:
                l = max(0, self.alphas[second_idx] +
                        self.alphas[first_idx] - self.c)
                h = min(
                    self.c, self.alphas[second_idx] + self.alphas[first_idx])
            if l == h:
                print('l == h')
                return 0

            # calculate eta, eta = 2 * k(1, 2) - k(1, 1) - k(1, 2) and eta < 0
            eta = 2.0 * self.kernel_trans[first_idx, second_idx] - self.kernel_trans[first_idx, first_idx] - self.kernel_trans[second_idx, second_idx]
            if eta >= 0:
                print('eta >= 0')
                return 0

            # update the second variable
            self.alphas[second_idx] -= self.label_matrix[second_idx] * \
                (error_first_idx - error_second_idx) / eta
            self.alphas[second_idx] = self.clip_alpha(
                self.alphas[second_idx], l, h)

            # update the error of the sample of second variables
            self.update_error_index(second_idx)

            # if the second_idx alpha is not changed enough, then do not update the first_idx alpha
            if np.abs(self.alphas[second_idx] - alpha_j_old) < 0.0001:
                print('j not moving enough')
                return 0

            # update the first_idx alpha
            self.alphas[first_idx] += self.label_matrix[second_idx] * \
                self.label_matrix[first_idx] * \
                (alpha_j_old - self.alphas[second_idx])

            # calculate the value of bias
            bias_1 = self.bias - error_first_idx - self.label_matrix[first_idx] * (self.alphas[first_idx] - alpha_i_old) * self.kernel_trans[first_idx, first_idx] - self.label_matrix[second_idx] * (self.alphas[second_idx] - alpha_j_old) * self.kernel_trans[first_idx, second_idx]
            bias_2 = self.bias - error_second_idx - self.label_matrix[first_idx] * (self.alphas[first_idx] - alpha_i_old) * self.kernel_trans[first_idx, second_idx] - self.label_matrix[second_idx] * (self.alphas[second_idx] - alpha_j_old) * self.kernel_trans[second_idx, second_idx]
            if (self.alphas[first_idx] > 0) and (self.alphas[first_idx] < self.c):
                self.bias = bias_1
            else:
                self.bias = (bias_1 + bias_2) / 2.0
            return 1
        else:
            return 0

    def platt_smo(self, max_iter):
        """
        train the svm classifier through platt-SMO 
        :param max_iter: int, the max iteration times
        :param k_tup: str, kernel function
        """
        iters = 0
        entire_set = True
        alpha_pair_changed = 0
        while (iters < max_iter) and ((alpha_pair_changed > 0) or entire_set):
            alpha_pair_changed = 0
            if entire_set:
                for param_idx in range(self.num_data):
                    alpha_pair_changed += self.inner_loop(param_idx)
                    print('full set, iters: %d i: %d, pairs changed %d' %
                          (iters, param_idx, alpha_pair_changed))
                iters += 1
            else:
                non_bound_is = np.nonzero(
                    (self.alphas.A > 0) * (self.alphas.A < self.c))[0]
                for param_idx in non_bound_is:
                    alpha_pair_changed += self.inner_loop(param_idx)
                    print('non-bound, iter: %d i:%d, pairs changed %d' %
                          (iters, param_idx, alpha_pair_changed))
                iters += 1
            if entire_set:
                entire_set = False
            elif alpha_pair_changed == 0:
                entire_set = True
            print('iteration number: %d' % iters)

    def predict(self, sample):
        kernel_eval = self.kernel(self.suport_vectors, sample)
        predict = kernel_eval.T * np.multiply(self.suport_labels, self.suport_alphas) + self.bias
        return np.sign(predict)

    def fit(self, max_iter):
        self.platt_smo(max_iter)
        suport_vectors_idx = np.nonzero(self.alphas > 0)[0]
        self.suport_alphas = self.alphas[suport_vectors_idx]
        self.suport_vectors = self.data_matrix[suport_vectors_idx]
        self.suport_labels = self.label_matrix[suport_vectors_idx]
        print('there are %d suport vectors' % np.shape(self.suport_vectors)[0])


def load(filename):
    data_matrix = []
    label_matrix = []
    fr = open(filename)
    for line in fr.readlines():
        line_array = line.strip().split('\t')
        data_matrix.append([float(line_array[0]), float(line_array[1])])
        label_matrix.append(float(line_array[2]))
    return data_matrix, label_matrix

# data_arr, label_arr = load('testSetRBF.txt')
# data_mat = np.matrix(data_arr)
# label_mat = np.matrix(np.matrix(label_arr).T)
# svm = SVM(data_mat, label_mat, 200, 0.0001, ('rbf', 1.3))
# svm.fit(10000)
#
# m, n = np.shape(data_mat)
# error_count = 0
# for i in range(svm.num_data):
#     pred = svm.predict(data_mat[i, :])
#     if np.sign(pred) != np.sign(label_arr[i]):
#         error_count += 1
# print('the training error rate is: %f' % (np.float(error_count / svm.num_data)))
#
# data_arr, label_arr = load('testSetRBF2.txt')
# data_mat = np.matrix(data_arr)
# label_mat = np.matrix(np.matrix(label_arr).T)
#
# m, n = np.shape(data_mat)
# error_count = 0
# for i in range(svm.num_data):
#     pred = svm.predict(data_mat[i, :])
#     if np.sign(pred) != np.sign(label_arr[i]):
#         error_count += 1
# print('the training error rate is: %f' % (np.float(error_count / svm.num_data)))

