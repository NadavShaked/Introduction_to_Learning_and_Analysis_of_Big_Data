import numpy as np
from scipy.spatial import distance


class Sample_details:
    def __init__(self, dist, label):
        self.distance = dist
        self.label = label


class Max_heap(object):
    # Array Implementation of max Heap
    # parent of a node is (i-1)/2 and child of a node is 2*i+1 and 2*i+2
    heap_type = 'MaxHeap'

    def __init__(self, capacity: int):
        self.arr = [None for x in range(capacity)]
        self.count = 0
        self.capacity = capacity

    def parent_node(self, i):
        if i < 0 or i > self.count:
            return False
        return int((i - 1) / 2)

    def left_child(self, i):
        left = 2 * i + 1
        return False if left >= self.count else left

    def right_child(self, i):
        right = 2 * i + 2
        return False if right >= self.count else right

    def insert(self, key: Sample_details):
        # count have number of element in array
        # so index of last element in heap is count-1
        if self.count == self.capacity:
            if self.arr[0].distance > key.distance:
                self.drop_max()
            else:
                return
        self.arr[self.count] = key
        self.heapify_up(self.count)
        self.count += 1

    def print_heap(self):
        print(', '.join([str(x.distance) for x in self.arr[:self.count]]))

    def heapify_down(self, parent):
        '''
        :param parent:
        heapy parant node from top to bottom
        '''
        left = self.left_child(parent)
        right = self.right_child(parent)

        if left and self.arr[left].distance > self.arr[parent].distance:
            max_ = left
        else:
            max_ = parent
        if right and self.arr[right].distance > self.arr[max_].distance:
            max_ = right

        if max_ != parent:
            # swap max index with parent
            self.arr[parent], self.arr[max_] = self.arr[max_], self.arr[parent]

            # recursive heapify
            self.heapify_down(max_)

    def heapify_up(self, child):
        parent = self.parent_node(child)

        if self.arr[parent].distance < self.arr[child].distance:
            # swap
            self.arr[parent], self.arr[child] = self.arr[child], self.arr[parent]
            self.heapify_up(parent)

    def drop_max(self):
        '''
        this is a max heap so root node is max, drop it replace it with last node.
        delete lst node then heapify top to bottom
        :return: max element of heap
        '''
        if self.count == 0:
            return
        max_data = self.arr[0]
        self.arr[0] = self.arr[self.count - 1]
        self.arr[self.count - 1] = None
        self.count -= 1
        self.heapify_down(0)
        return max_data


class KNN_classifier:
    def __init__(self, k: int, x_train: np.array, y_train: np.array):
        """
        :param k: value of the nearest neighbour parameter k
        :param x_train: numpy array of size (m, d) containing the training sample
        :param y_train: numpy array of size (m, 1) containing the labels of the training sample
        :return: classifier data structure
        """
        self.k = k
        self.number_of_samples, self.sample_dim = x_train.shape
        self.Xs = x_train
        self.Ys = y_train

    def predict(self, sample: np.array):
        """
        :param sample: numpy array of size (1, d) represent sample
        :return: classified label
        """
        k_heap = Max_heap(self.k)

        # insert nearest neighbors samples to heap
        for i in range(self.number_of_samples):
            dist = distance.euclidean(self.Xs[i], sample)
            details = Sample_details(dist, self.Ys[i])
            k_heap.insert(details)

        # insert nearest neighbors to dict and count them
        label_dict = {}
        key: Sample_details = k_heap.drop_max()
        while key is not None:
            if key.label in label_dict:
                label_dict[key.label] = label_dict[key.label] + 1
            else:
                label_dict[key.label] = 1
            key: Sample_details = k_heap.drop_max()

        most_common_label = -1
        most_common_value = -1
        for key, value in label_dict.items():
            if value > most_common_value:
                most_common_label = key
                most_common_value = value

        return most_common_label


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """
    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return KNN_classifier(k, x_train, y_train)


def predictknn(classifier: KNN_classifier, x_test: np.array):
    """
    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    sample_count = x_test.shape[0]
    Ytestprediction = np.zeros(sample_count)
    for i in range(sample_count):
        Ytestprediction[i] = classifier.predict(x_test[i])

    return Ytestprediction.reshape([sample_count, 1])


def simple_test():
    # data = np.load('mnist_all_not_compressed.npz')
    data = np.load('ex1/mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifier = learnknn(5, x_train, y_train)

    preds = predictknn(classifier, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def simple_test_from_HW():
    k = 1
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 0, 1])
    classifier = learnknn(k, x_train, y_train)
    x_test = np.array([[10, 11], [3.1, 4.2], [2.9, 4.2], [5, 6]])
    y_testprediction = predictknn(classifier, x_test)
    return y_testprediction


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
