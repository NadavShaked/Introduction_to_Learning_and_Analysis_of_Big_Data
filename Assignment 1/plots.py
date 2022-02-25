import random
import numpy as np
import matplotlib.pyplot as plt
from ex1.nearest_neighbour import gensmallm, learnknn, predictknn


def question_2a(test_sample_size_array, k=1):
    number_of_returns = 10

    data = np.load('mnist_all.npz')

    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    average_error = np.zeros(len(test_sample_size_array))
    max_error = np.zeros(len(test_sample_size_array))
    min_error = np.zeros(len(test_sample_size_array))
    for training_sample_size in range(1, len(test_sample_size_array) + 1):
        test_sample_arr_size = len(test1) + len(test3) + len(test4) + len(test6)
        empirical_error_array = np.zeros(number_of_returns)
        x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6],
                                   test_sample_arr_size)
        for i in range(number_of_returns):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6],
                                         test_sample_size_array[training_sample_size - 1])
            classifier = learnknn(k, x_train, y_train)
            y_testpredict = predictknn(classifier, x_test)
            empirical_error_array[i] = np.mean(y_test != y_testpredict.reshape(1, y_testpredict.shape[0]))
        average_error[training_sample_size - 1] = np.average(empirical_error_array)
        max_error[training_sample_size - 1] = empirical_error_array.max()
        min_error[training_sample_size - 1] = empirical_error_array.min()

    return average_error, max_error, min_error


def question_2a_generate_plot():
    test_sample_size_array = [10, 25, 50, 75, 100]

    average_error, max_error, min_error = question_2a(test_sample_size_array)
    upper_error = max_error - average_error
    lower_error = average_error - min_error

    fig = plt.figure()
    yerr = [upper_error, lower_error]
    plt.errorbar(test_sample_size_array, average_error, yerr=yerr, label='k = 1')
    plt.legend(loc='upper right')
    plt.title('Average Error as a Function of Sample Size')
    plt.xlabel("Training Sample Sizes")
    plt.ylabel("Average Prediction Error")

    plt.show()


def question_2e(k_max=11, test_sample_size=100):
    number_of_repeats = 10

    data = np.load('mnist_all.npz')

    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    test_sample_arr_size = len(test1) + len(test3) + len(test4) + len(test6)
    x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6],
                               test_sample_arr_size)
    average_error = np.zeros(k_max)
    max_error = np.zeros(k_max)
    min_error = np.zeros(k_max)
    for k in range(1, k_max + 1):
        empirical_error_array = np.zeros(number_of_repeats)
        for i in range(number_of_repeats):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], test_sample_size)
            classifier = learnknn(k, x_train, y_train)
            y_testpredict = predictknn(classifier, x_test)
            empirical_error_array[i] = np.mean(y_test != y_testpredict.reshape(1, y_testpredict.shape[0]))
        average_error[k - 1] = np.average(empirical_error_array)
        max_error[k - 1] = empirical_error_array.max()
        min_error[k - 1] = empirical_error_array.min()

    return average_error, max_error, min_error


def question_2e_generate_plot():
    k = 11

    average_error, max_error, min_error = question_2e(k)
    upper_error = max_error - average_error
    lower_error = average_error - min_error

    fig = plt.figure()
    yerr = [upper_error, lower_error]
    plt.errorbar(range(1, k + 1), average_error, yerr=yerr, label='repeats = 10')
    plt.legend(loc='upper left')
    plt.title('Average Error as a Function of k')
    plt.xlabel("k nearest neighbors")
    plt.ylabel("Average Prediction Error")

    plt.show()


def question_2f(k_max=11, test_sample_size=100):
    number_of_repeats = 10

    data = np.load('mnist_all.npz')

    train1 = data['train1']
    train3 = data['train3']
    train4 = data['train4']
    train6 = data['train6']

    test1 = data['test1']
    test3 = data['test3']
    test4 = data['test4']
    test6 = data['test6']

    average_error = np.zeros(k_max)
    max_error = np.zeros(k_max)
    min_error = np.zeros(k_max)
    for k in range(1, k_max + 1):
        empirical_error_array = np.zeros(number_of_repeats)
        for i in range(number_of_repeats):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], test_sample_size)
            y_train = corrupt_labels(y_train, 20, np.array([1, 3, 4, 6]))

            test_sample_arr_size = len(test1) + len(test3) + len(test4) + len(test6)
            x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6],
                                       test_sample_arr_size)
            y_test = corrupt_labels(y_test, 20, np.array([1, 3, 4, 6]))

            classifier = learnknn(k, x_train, y_train)
            y_testpredict = predictknn(classifier, x_test)
            empirical_error_array[i] = np.mean(y_test != y_testpredict.reshape(1, y_testpredict.shape[0]))
        average_error[k - 1] = np.average(empirical_error_array)
        max_error[k - 1] = empirical_error_array.max()
        min_error[k - 1] = empirical_error_array.min()

    return average_error, max_error, min_error


def question_2f_generate_plot():
    k = 11

    average_error, max_error, min_error = question_2e(k)
    upper_error = max_error - average_error
    lower_error = average_error - min_error

    fig = plt.figure()
    yerr = [upper_error, lower_error]
    plt.errorbar(range(1, k + 1), average_error, yerr=yerr, label='repeats = 10')
    plt.legend(loc='upper left')
    plt.title('Average Error as a Function of k')
    plt.xlabel("k nearest neighbors")
    plt.ylabel("Average Prediction Error")

    plt.show()


def corrupt_labels(labels_array, percentage_of_array, labels):
    number_of_items_to_corrupt = len(labels_array) // percentage_of_array
    for i in range(number_of_items_to_corrupt):
        labels_array[i] = random.choice(np.setdiff1d(labels, [labels_array[i]]))

    return labels_array


if __name__ == '__main__':
    question_2a_generate_plot()
    question_2e_generate_plot()
    question_2f_generate_plot()
