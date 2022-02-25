import numpy as np
import matplotlib.pyplot as plt


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


def bayeslearn(x_train: np.array, y_train: np.array):
    """

    :param x_train: 2D numpy array of size (m, d) containing the the training set. The training samples should be binarized
    :param y_train: numpy array of size (m, 1) containing the labels of the training set
    :return: a triple (allpos, ppos, pneg) the estimated conditional probabilities to use in the Bayes predictor
    """

    d = x_train.shape[1]
    ppos = np.zeros(d)
    pneg = np.zeros(d)
    allpos = np.mean(y_train == 1)  # P[Y = 1]

    for i in range(d):
        x_i_predict_and_y_array = np.concatenate((x_train[:, i].reshape(-1, 1), y_train.reshape(-1, 1)), axis=1)

        # P[X(i) = 1 and Y = 1]
        x_eq_1_and_y_eq_1_row = [1, 1]
        p_x_eq_1_and_y_eq_1 = np.mean(np.apply_along_axis(lambda x: np.array_equal(x, x_eq_1_and_y_eq_1_row), 1, x_i_predict_and_y_array))
        # P[X(i) = 1 | Y = 1]
        ppos[i] = p_x_eq_1_and_y_eq_1 / allpos

        # P[X(i) = 1 and Y = -1]
        x_eq_1_and_y_eq_minus_1_row = [1, -1]
        p_x_eq_1_and_y_eq_minus_1 = np.mean(np.apply_along_axis(lambda x: np.array_equal(x, x_eq_1_and_y_eq_minus_1_row), 1, x_i_predict_and_y_array))
        # P[X(i) = 1 | Y = -1]
        pneg[i] = p_x_eq_1_and_y_eq_minus_1 / (1 - allpos)

    return allpos, ppos, pneg


def bayespredict(allpos: float, ppos: np.array, pneg: np.array, x_test: np.array):
    """

    :param allpos: scalar between 0 and 1, indicating the fraction of positive labels in the training sample
    :param ppos: numpy array of size (d, 1) containing the empirical plug-in estimate of the positive conditional probabilities
    :param pneg: numpy array of size (d, 1) containing the empirical plug-in estimate of the negative conditional probabilities
    :param x_test: numpy array of size (n, d) containing the test samples
    :return: numpy array of size (n, 1) containing the predicted labels of the test samples
    """

    m, d = x_test.shape
    predict = np.zeros(m)
    for i in range(m):
        p_y_eq_1 = np.log(allpos)
        p_y_eq_minus_1 = np.log(1 - allpos)
        for j in range(d):
            if x_test[i][j] == 1:
                p_y_eq_1 += np.log(ppos[j])
                p_y_eq_minus_1 += np.log(pneg[j])
            else:
                p_y_eq_1 += np.log(1 - ppos[j])
                p_y_eq_minus_1 += np.log(1 - pneg[j])

        if p_y_eq_1 == p_y_eq_minus_1:
            continue    # keep predict[i] = 0
        predict[i] = 1 if p_y_eq_1 > p_y_eq_minus_1 else -1

    return predict.reshape(-1, 1)


def simple_test():
    # load sample data from question 2, digits 3 and 5 (this is just an example code, don't forget the other part of
    # the question)
    data = np.load('mnist_all.npz')

    train3 = data['train3']
    train5 = data['train5']

    test3 = data['test3']
    test5 = data['test5']

    m = 500
    n = 50
    d = train3.shape[1]

    x_train, y_train = gensmallm([train3, train5], [-1, 1], m)

    x_test, y_test = gensmallm([test3, train5], [-1, 1], n)

    # threshold the images (binarization)
    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    x_test = np.where(x_test > threshold, 1, 0)

    # run naive bayes algorithm
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    assert isinstance(ppos, np.ndarray) \
           and isinstance(pneg, np.ndarray), "ppos and pneg should be numpy arrays"

    assert 0 <= allpos <= 1, "allpos should be a float between 0 and 1"

    y_predict = bayespredict(allpos, ppos, pneg, x_test)

    assert isinstance(y_predict, np.ndarray), "The output of the function bayespredict should be numpy arrays"
    assert y_predict.shape == (n, 1), f"The output of bayespredict should be of size ({n}, 1)"

    print(f"Prediction error = {np.mean(y_test != y_predict)}")


def convert_data_set(set: np.array):
    threshold = 128
    return np.where(set > threshold, 1, 0)


def question_2_a_func(first_digit_train_set, second_digit_train_set, first_digit_test_set, second_digit_test_set, training_size_array):
    error_array = np.zeros(10)

    for i, m in enumerate(training_size_array):
        x_train, y_train = gensmallm([first_digit_train_set, second_digit_train_set], [-1, 1], m)
        x_train = convert_data_set(x_train)
        allpos, ppos, pneg = bayeslearn(x_train, y_train)

        x_test, y_test = gensmallm([first_digit_test_set, second_digit_test_set], [-1, 1], first_digit_test_set.shape[0] + second_digit_test_set.shape[0])
        x_test = convert_data_set(x_test)
        y_predict = bayespredict(allpos, ppos, pneg, x_test)

        error_array[i] = np.mean(y_test.reshape(-1, 1) != y_predict)

    return error_array


def question_2_a():
    data = np.load('mnist_all.npz')

    training_size_array = np.array(range(1000, 10001, 1000))
    error_array_for_digits_0_and_1 = question_2_a_func(0, 1, data['train0'], data['train1'], data['test0'], data['test1'], training_size_array)
    error_array_for_digits_3_and_5 = question_2_a_func(3, 5, data['train3'], data['train5'], data['test3'], data['test5'], training_size_array)

    fig = plt.figure()
    plt.plot(training_size_array, error_array_for_digits_0_and_1, label='digits 0 and 1')
    plt.plot(training_size_array, error_array_for_digits_3_and_5, label='digits 3 and 5')
    plt.legend(loc='upper right')
    plt.title(f'Average Error as a Function of Training Sample Size')
    plt.xlabel("training sample size")
    plt.ylabel("Average Prediction Error")

    plt.show()


def question_2_c(sample_size: int):
    data = np.load('mnist_all.npz')

    x_train, y_train = gensmallm([data['train0'], data['train1']], [-1, 1], sample_size)
    x_train = convert_data_set(x_train)
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    fig = plt.figure()
    plt.imshow(ppos.reshape(28, 28), cmap='hot')
    plt.show()

    plt.imshow(pneg.reshape(28, 28), cmap='hot')
    plt.show()


def question_2_d_func(first_digit: int, second_digit: int, first_digit_train_set, second_digit_train_set, first_digit_test_set, second_digit_test_set, m: int, allpos_fake_value: float):
    x_train, y_train = gensmallm([first_digit_train_set, second_digit_train_set], [-1, 1], m)
    x_train = convert_data_set(x_train)
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    x_test, y_test = gensmallm([first_digit_test_set, second_digit_test_set], [-1, 1], first_digit_test_set.shape[0] + second_digit_test_set.shape[0])
    x_test = convert_data_set(x_test)

    y_predict_real = bayespredict(allpos, ppos, pneg, x_test)
    y_predict_fake = bayespredict(allpos_fake_value, ppos, pneg, x_test)

    real_error = np.mean(y_test.reshape(-1, 1) != y_predict_real)
    fake_error = np.mean(y_test.reshape(-1, 1) != y_predict_fake)

    test_size = y_predict_real.shape[0]
    changed_from_1_to_minus1_count = 0
    changed_from_minus1_to_1_count = 0
    for i in range(test_size):
        if y_predict_real[i][0] != y_predict_fake[i][0]:
            if y_predict_real[i][0] == 1:
                changed_from_1_to_minus1_count += 1
            else:
                changed_from_minus1_to_1_count += 1

    changed_from_1_to_minus1 = changed_from_1_to_minus1_count / test_size
    changed_from_minus1_to_1 = changed_from_minus1_to_1_count / test_size

    #print(f'For digits {first_digit} and {second_digit}:the real(allpos = {allpos}) error is {real_error}, and the fake(allpos = {allpos_fake_value}) error is {fake_error}, changed from 1 to -1 is {changed_from_1_to_minus1}%, and changed from -1 to 1 is {changed_from_minus1_to_1}%')


def question_2_d(sample_size: int, allpos_fake_value: float):
    data = np.load('mnist_all.npz')

    question_2_d_func(0, 1, data['train0'], data['train1'], data['test0'], data['test1'], sample_size, allpos_fake_value)
    question_2_d_func(3, 5, data['train3'], data['train5'], data['test3'], data['test5'], sample_size, allpos_fake_value)


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    # question_2_a()
    # question_2_c()
    #question_2_d(10000, 0.75)
