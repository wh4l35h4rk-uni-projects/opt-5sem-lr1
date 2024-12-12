import numpy as np
import random


def f(x, A, b):
    mul1 = np.matmul(np.matmul(x.T, A), x)
    mul2 = np.matmul(b, x)
    return mul1 * 0.5 + mul2


def df(x, A, b):
    return np.matmul(A, x) + b.T


def grad_desc(x0, A, b):
    e = 1e-6
    step = 1e-4
    n = 0

    while True:
        n += 1
        x1 = x0 - df(x0, A, b) * step
        if np.linalg.norm(x1 - x0) < e:
            break
        x0 = x1
    return x0, n


if __name__ == '__main__':
    random.seed(3)
    n = 6
    x0 = np.array([[random.randint(-100, 100) for i in range(1)] for j in range(n)])
    b = np.array([[random.randint(-100, 100) for i in range(n)] for j in range(1)])

    while True:
        A = np.array([[random.randint(-100, 100) for i in range(n)] for j in range(n)])
        A = A + A.T
        if all([e > 0 for e in np.linalg.eigvals(A)]):
            print(np.linalg.eigvals(A))
            break

    print(A)
    print("x0:", x0.T)
    print("b:", b)
    print(grad_desc(x0, A, b))
