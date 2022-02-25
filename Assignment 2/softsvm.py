import numpy as np
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt

isDebugMode = True


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def createU(d: int, m: int):
    """

    :param d: the dimension of sample
    :param m: number of sample, containing the training sample
    :return: u vector, a numpy array of size (1, d + m)
    """
    return matrix(np.concatenate((np.zeros(d), np.full([1, m], 1 / m)), axis=None))


def createV(m: int):
    """

    :param m: number of sample, containing the training sample
    :return: v vector, a numpy array of size (1, 2 * m)
    """
    return matrix(np.concatenate((np.zeros(m), np.ones(m)), axis=None))


def createH(d: int, m: int, l):
    """

    :param d: the dimension of sample
    :param m: number of sample, containing the training sample
    :param l: the parameter lambda of the soft SVM algorithm
    :return: H metrix, a numpy array of size (d + m, d + m)
    """
    H = np.zeros((d + m, d + m))
    for i in range(d):
        H[i][i] = 2 * l

    return matrix(H)


def createA(trainX: np.array, trainy: np.array):
    """

    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: A metrix, a numpy array of size (2 * m, d + m)
    """
    m, d = trainX.shape

    A = np.zeros((2 * m, d + m))

    # top right block of size m x m
    for row, col in enumerate(range(d, m + d)):
        A[row][col] = 1

    # bottom right block of size m x m
    for row, col in enumerate(range(d, m + d)):
        A[m + row][col] = 1

    # bottom left block of size m x d
    constraints = trainX * trainy.reshape(-1, 1)
    for row in range(m):
        for col in range(d):
            A[m + row][col] = constraints[row][col]

    return matrix(A)


def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    sample_size, sample_dim = trainX.shape
    u = createU(sample_dim, sample_size)
    v = createV(sample_size)
    H = createH(sample_dim, sample_size, l)
    A = createA(trainX, trainy)

    sol = solvers.qp(H, u, -A, -v)
    w = np.array(sol["x"])[:sample_dim]

    return w


def k_fold_cross_validation(k: int, lambda_array: np.array, trainX: np.array, trainy: np.array):
    min_error = 9223372036854775807
    best_lambda = None

    # generate k random sets
    indices = np.random.permutation(trainX.shape[0])
    split_indices = np.array(np.split(indices, k))
    splitted_sample_sets = [None] * k
    splitted_label_sets = [None] * k
    for i in range(k):
        splitted_sample_sets[i] = trainX[split_indices[i]]
        splitted_label_sets[i] = trainy[split_indices[i]]

    for l in lambda_array:
        error_sum = 0
        for i in range(k):
            if isDebugMode:
                print(f"lambda: {l}, i: {i}")

            # gernerate validation set training sample set
            V_x = splitted_sample_sets[i]
            V_y = splitted_label_sets[i]
            X_set_list = [set for set_index, set in enumerate(splitted_sample_sets) if set_index != i]
            Y_set_list = [set for set_index, set in enumerate(splitted_label_sets) if set_index != i]
            Sx_tag = X_set_list[0]
            Sy_tag = Y_set_list[0]
            for set_index in range(1, len(X_set_list)):
                Sx_tag = np.concatenate((Sx_tag, X_set_list[set_index]), axis=0)
                Sy_tag = np.concatenate((Sy_tag, Y_set_list[set_index]), axis=0)

            w = softsvm(l, Sx_tag, Sy_tag)

            #calculate the error on validation test
            validation_set_predicty = np.sign(V_x @ w)
            error = np.mean(V_y != validation_set_predicty.reshape(1, validation_set_predicty.shape[0]))

            error_sum += error

        avg_error = error_sum / k

        if isDebugMode:
            print(f"lambda: {l}, avg_error: {avg_error}")

        if avg_error < min_error:
            min_error = avg_error
            best_lambda = l

    return best_lambda


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


def question2(m: int, l_values):
    number_of_returns = 10

    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    train_average_error = np.zeros(len(l_values))
    train_max_error = np.zeros(len(l_values))
    train_min_error = np.zeros(len(l_values))
    test_average_error = np.zeros(len(l_values))
    test_max_error = np.zeros(len(l_values))
    test_min_error = np.zeros(len(l_values))
    
    # run the softsvm algorithm
    for i, l in enumerate(l_values):
        train_empirical_error_array = np.zeros(number_of_returns)
        test_empirical_error_array = np.zeros(number_of_returns)
        for j in range(10):
            # Get a random m training examples from the training set
            indices = np.random.permutation(trainX.shape[0])
            _trainX = trainX[indices[:m]]
            _trainy = trainy[indices[:m]]

            w = softsvm(10 ** l, _trainX, _trainy)
            train_predicty = np.sign(_trainX @ w)
            train_empirical_error_array[j] = np.mean(_trainy != train_predicty.reshape(1, train_predicty.shape[0]))
            test_predicty = np.sign(testX @ w)
            test_empirical_error_array[j] = np.mean(testy != test_predicty.reshape(1, test_predicty.shape[0]))
        train_average_error[i] = np.average(train_empirical_error_array)
        train_max_error[i] = train_empirical_error_array.max()
        train_min_error[i] = train_empirical_error_array.min()
        test_average_error[i] = np.average(test_empirical_error_array)
        test_max_error[i] = test_empirical_error_array.max()
        test_min_error[i] = test_empirical_error_array.min()

    return train_average_error, train_max_error, train_min_error, test_average_error, test_max_error, test_min_error


def question2_a():
    l_values = range(1, 11)

    train_average_error, train_max_error, train_min_error, test_average_error, test_max_error, test_min_error = question2(100, l_values)

    train_upper_error = test_max_error - test_average_error
    train_lower_error = test_average_error - test_min_error

    test_upper_error = test_max_error - test_average_error
    test_lower_error = test_average_error - test_min_error

    fig = plt.figure()
    train_yerr = [train_upper_error, train_lower_error]
    plt.errorbar(l_values, train_average_error, yerr=train_yerr, label='train set', capsize=4)
    test_yerr = [test_upper_error, test_lower_error]
    plt.errorbar(np.array(range(1, 11)) + 0.1, test_average_error, yerr=test_yerr, label='test set', capsize=4)
    plt.legend(loc='upper right')
    plt.title('Average Error as a Function of λ = 10^k')
    plt.xlabel("k size")
    plt.ylabel("Average Prediction Error")

    plt.show()


def question2_b():
    l_values = np.array([1, 3, 5, 8])
    train_average_error, train_max_error, train_min_error, test_average_error, test_max_error, test_min_error = question2(100, l_values)

    train_upper_error = train_max_error - train_average_error
    train_lower_error = train_average_error - train_min_error

    test_upper_error = test_max_error - test_average_error
    test_lower_error = test_average_error - test_min_error

    fig = plt.figure()
    train_yerr = [train_upper_error, train_lower_error]
    plt.errorbar(l_values, train_average_error, yerr=train_yerr, label='train set', fmt='o', capsize=4)
    test_yerr = [test_upper_error, test_lower_error]
    plt.errorbar(np.array(l_values) + 0.1, test_average_error, yerr=test_yerr, label='test set', fmt='o', capsize=4)
    plt.legend(loc='lower right')
    plt.title('Average Error as a Function of λ = 10^k')
    plt.xlabel("k size")
    plt.ylabel("Average Prediction Error")

    plt.show()


def question4_b():
    lambda_array = np.array([1, 10, 100])

    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    best_lambda = k_fold_cross_validation(5, lambda_array, trainX, trainy)

    # run soft-SVM with the best parameter
    w = softsvm(best_lambda, trainX, trainy)

    # calculate the error on test set
    test_predicty = np.sign(testX @ w)
    error = np.mean(trainy != test_predicty.reshape(1, test_predicty.shape[0]))

    if isDebugMode:
        print(f"test set - best_lambda: {best_lambda}, error: {error}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    # question2_a()
    # question2_b()
    # question4_b()
