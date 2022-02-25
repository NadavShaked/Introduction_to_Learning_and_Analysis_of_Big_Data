import numpy as np


def isMatrixOrthonormal(A):
    d = A.shape[1]
    for i in range(d - 1):
        norm = np.linalg.norm(A[:, i])
        if not np.isclose(norm, 1):
            return False
        for j in range(i + 1, d):
            product = np.sum(A[:, i] * A[:, j])
            if not np.isclose(product, 0):
                return False
    return True


def question7(X: np.array, k: int):
    m, d = X.shape
    A = X.T @ X
    eigh = np.linalg.eigh(A)
    eigenvalues = eigh[0]
    eigenvectors = eigh[1]

    distortionByDefinition = np.sum(eigenvalues[:d - k])
    U = eigenvectors[:, d - k: d]

    # question 7a
    print("question 7a")
    print(f"distortion by definition is {distortionByDefinition}")
    print()

    # question 7b
    print("question 7b")
    print(f"U^T is {'' if isMatrixOrthonormal(U) else 'not '}orthonormal")
    print(U.T)
    print()

    # question 7c
    print("question 7c")
    distortionByValues = 0
    for i in range(m):
        x = X[i]
        restoredVector = U @ U.T @ x
        print(f"restored example {i} is {restoredVector}, origin example is {x}")

        distortionByValues += pow(np.linalg.norm(x - restoredVector), 2)

    print(f"distortion by value is {distortionByValues}")


if __name__ == '__main__':
    X = np.array([[1, -2, 5, 4], [3, 2, 1, -5], [-10, 1, -4, 6]])
    k = 2  # reduce to dimension

    question7(X, k)
