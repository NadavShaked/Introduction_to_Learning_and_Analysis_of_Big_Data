import numpy as np
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt
from matplotlib import colors

isDebugMode = True


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def classifyBySign(x, trainX: np.array, sigma: float, alpha: np.array):
    m = trainX.shape[0]
    sum = 0
    for i in range(m):
        x_i = trainX[i]
        sum += alpha[i] * gaussianKernel(x, x_i, sigma)

        sign = np.sign(sum)

    return sign


def gaussianKernel(x1, x2, sigma: float):
    """

    :param x1: sample, containing in the training sample
    :param x2: sample, containing in the training sample
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :return: K_x1_x2, gaussian kernel value, K(x, x')
    """
    x = x1 - x2
    power = - (pow(np.linalg.norm(x), 2) / (2 * sigma))
    K_x1_x2 = np.exp(power)
    return K_x1_x2


def createG(trainX: np.array, sigma: float):
    """

    :param trainX: numpy array of size (m, d) containing the training sample
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :return: G, Gram matrix of train set by gaussianKernel
    """
    m = trainX.shape[0]
    G = np.zeros([m, m])
    for i in range(m):
        for j in range(i, m):
            x1 = trainX[i]
            x2 = trainX[j]
            G[i][j] = gaussianKernel(x1, x2, sigma)
            G[j][i] = G[i][j]

    return G


def createU(m: int):
    """

    :param m: number of sample, containing the training sample
    :return: u vector, a numpy array of size (1, 2 * m)
    """
    return matrix(np.concatenate((np.zeros(m), np.full([1, m], 1 / m)), axis=None))


def createV(m: int):
    """

    :param m: number of sample, containing the training sample
    :return: v vector, a numpy array of size (1, 2 * m)
    """
    return matrix(np.concatenate((np.zeros(m), np.ones(m)), axis=None))


def createH(G: np.array, l):
    """

    :param G: Gram matrix of train set
    :param l: the parameter lambda of the soft SVM algorithm
    :return: H metrix, a numpy array of size (2 * m, 2 * m)
    """
    m = G.shape[0]
    H = np.zeros((2 * m, 2 * m))

    top_right_block = 2 * l * G

    for i in range(m):
        for j in range(m):
            H[i][j] = top_right_block[i][j]

    epsilon = pow(10, -4)
    if np.linalg.eigvals(H).min() < 0:
        H += np.identity(2 * m) * epsilon

    return matrix(H)


def createA(G: np.array, trainy: np.array):
    """

    :param G: Gram matrix of train set
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: A metrix, a numpy array of size (2 * m, 2 * m)
    """
    m = G.shape[0]

    A = np.zeros((2 * m, 2 * m))

    # top right block of size m x m
    for row, col in enumerate(range(m, 2 * m)):
        A[row][col] = 1

    # bottom right block of size m x m
    for row, col in enumerate(range(m, 2 * m)):
        A[m + row][col] = 1

    # bottom right block of size m x m
    constraints = G * trainy.reshape(-1, 1)
    for row in range(m):
        for col in range(m):
            A[m + row][col] = constraints[row][col]

    return matrix(A)


def softsvmbf(l: float, sigma: float, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    G = createG(trainX, sigma)

    sample_size = G.shape[0]
    u = createU(sample_size)
    v = createV(sample_size)
    H = createH(G, l)
    A = createA(G, trainy)

    sol = solvers.qp(H, u, -A, -v)
    alpha = np.array(sol["x"])[:sample_size]

    return alpha


def k_fold_cross_validation(k: int, lambda_array: np.array, sigma_array: np.array, trainX: np.array, trainy: np.array):
    min_error = 9223372036854775807
    best_lambda = None
    best_sigma = None

    # generate k random sets
    indices = np.random.permutation(trainX.shape[0])
    split_indices = np.array(np.split(indices, k))
    splitted_sample_sets = [None] * k
    splitted_label_sets = [None] * k
    for i in range(k):
        splitted_sample_sets[i] = trainX[split_indices[i]]
        splitted_label_sets[i] = trainy[split_indices[i]]

    for l in lambda_array:
        for sigma in sigma_array:
            error_sum = 0
            for i in range(k):
                if isDebugMode:
                    print(f"lambda: {l}, sigma: {sigma}, i: {i}")

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

                alpha = softsvmbf(l, sigma, Sx_tag, Sy_tag)

                # calculate the error on validation test
                validation_set_size = V_x.shape[0]
                predict_y = np.zeros(validation_set_size)
                for i in range(validation_set_size):
                    predict_y[i] = classifyBySign(V_x[i], Sx_tag, sigma, alpha)

                error_sum += np.mean(V_y != predict_y.reshape(-1, 1))

            avg_error = error_sum / k

            if isDebugMode:
                print(f"lambda: {l}, sigma: {sigma}, avg_error: {avg_error}")

            if avg_error < min_error:
                min_error = avg_error
                best_lambda = l
                best_sigma = sigma

    return best_lambda, best_sigma


def simple_test():
    # load question 2 data
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvmbf(10, 0.1, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def question4_a():
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']

    points_with_classified_label = np.c_[trainX, trainy]
    positive_label_points = points_with_classified_label[
                                points_with_classified_label[:, points_with_classified_label.shape[1] - 1] > 0][:,
                            : points_with_classified_label.shape[1] - 1]
    negative_label_points = points_with_classified_label[
                                points_with_classified_label[:, points_with_classified_label.shape[1] - 1] < 0][:,
                            : points_with_classified_label.shape[1] - 1]
    plt.scatter(positive_label_points[:, 0], positive_label_points[:, 1])
    plt.scatter(negative_label_points[:, 0], negative_label_points[:, 1], color='red')
    plt.show()


def question4_b():
    lambda_array = np.array([1, 10, 100])
    sigma_array = np.array([0.01, 0.5, 1])

    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    best_lambda, best_sigma = k_fold_cross_validation(5, lambda_array, sigma_array, trainX, trainy)

    # run soft-SVM with the best parameters
    alpha = softsvmbf(best_lambda, best_sigma, trainX, trainy)

    # calculate the error on test set
    test_sample_size = testX.shape[0]
    predict_y = np.zeros(test_sample_size)
    for i in range(test_sample_size):
        predict_y[i] = classifyBySign(testX[i], trainX, best_sigma, alpha)

    mean = np.mean(testy != predict_y.reshape(-1, 1))
    if isDebugMode:
        print(f"test set - best_lambda: {best_lambda}, best_sigma: {best_sigma}, error: {mean}")


def question4_d():
    l = 100
    sigma_array = [0.01, 0.5, 1]
    points_in_axis = 200
    space_size = 8.5

    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']

    for sigma in sigma_array:
        alpha = softsvmbf(l, sigma, trainX, trainy)

        grid_linspace = np.linspace(-space_size, space_size, points_in_axis)
        grid = np.zeros([points_in_axis, points_in_axis])
        for row, x in enumerate(list(grid_linspace)):
            if isDebugMode:
                print(f'calculate row {row + 1} of {points_in_axis}')

            for col, y in enumerate(list(grid_linspace)):
                grid[row][col] = classifyBySign(np.array([x, y]), trainX, sigma, alpha)

        cs = plt.contourf(grid, cmap=colors.ListedColormap(['r', 'b']))
        cbar = plt.colorbar(cs)

        plt.show()
        plt.clf()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    # question4_a(simple_test)
    # question4_b()
    # question4_d()
