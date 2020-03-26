import numpy as np
import pandas as pd
import cvxopt
import cvxopt.solvers
from collections import Counter
from itertools import combinations_with_replacement
from time import time

#Load Data

xtr0 = pd.read_csv("data/Xtr0.csv", " ", header=0)
xtr1 = pd.read_csv("data/Xtr1.csv", " ", header=0)
xtr2 = pd.read_csv("data/Xtr2.csv", " ", header=0)
x_tr = np.append(np.append(xtr0, xtr1), xtr2)

xte0 = pd.read_csv("data/Xte0.csv", " ", header=0)
xte1 = pd.read_csv("data/Xte1.csv", " ", header=0)
xte2 = pd.read_csv("data/Xte2.csv", " ", header=0)
x_te = np.append(np.append(xte0, xte1), xte2)

ytr0 = pd.read_csv("data/Ytr0.csv", index_col=0, header=0)
ytr1 = pd.read_csv("data/Ytr1.csv", index_col=0, header=0)
ytr2 = pd.read_csv("data/Ytr2.csv", index_col=0, header=0)
y_train = np.append(np.append(ytr0, ytr1), ytr2)
y_train[y_train[:] == 0] = -1


def prepare_data(x, k):
    p = ['G', 'T', 'A', 'C', 'T', 'A', 'C', 'G', 'A', 'C', 'G', 'T', 'A', 'C', 'G', 'T', 'C', 'G', 'T', 'A']
    subsequence = []
    for i in combinations_with_replacement(p, k):
        subsequence.append(list(i))
    subsequence = np.asarray(subsequence)
    subsequence = np.unique(subsequence, axis=0)
    subsequence = ["".join(j) for j in subsequence[:, :].astype(str)]

    index = np.arange(0, len(subsequence))

    features = np.zeros((len(x), len(subsequence)))  # To store the occurence of each string
    for i in range(0, len(x)):
        s = x[i]
        c = [s[j:j + k] for j in range(len(s) - k + 1)]
        counter = Counter(c)
        j = 0
        for m in subsequence:
            features[i][j] = counter[m]
            j = j + 1

    features_array = features[:, index]
    features_array = features_array / np.max(np.abs(features_array), axis=0)

    return features_array


x_train = prepare_data(x_tr, k=4)
x_test = prepare_data(x_te, k=4)


def rbf_kernel(x, y, sigma=3):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


class SVM(object):
    def __init__(self, kernel=rbf_kernel, C=1):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):

        n_samples, n_features = X.shape
        # Computation of the gram matrix
        Gram = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            if (i % 100 == 0):
                print(i, "/", n_samples)
            for j in range(n_samples):
                Gram[i, j] = self.kernel(X[i], X[j])

                # Components for quadratic program problem
        P = cvxopt.matrix(np.outer(y, y) * Gram)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(np.zeros(1))

        temp1 = np.diag(np.ones(n_samples) * -1)
        temp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((temp1, temp2)))
        temp1 = np.zeros(n_samples)
        temp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((temp1, temp2)))

        # Solving quadratic progam problem - obtaining Lagrange multipliers
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers, threshold = 1e-6
        sup_vec = alphas > 1e-6
        ind = np.arange(len(alphas))[sup_vec]

        # Creating support vectors
        self.alphas = alphas[sup_vec]
        self.sup_vec = X[sup_vec]
        self.sup_vec_y = y[sup_vec]

        # Fitting support vectors with the intercept
        self.b = 0
        for i in range(len(self.alphas)):
            self.b += self.sup_vec_y[i]
            self.b -= np.sum(self.alphas * self.sup_vec_y * Gram[ind[i], sup_vec])
        self.b /= len(self.alphas)
        print(self.b)

        # Weight for rbf kernel
        self.w = None

    def predict(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for alphas, sup_vec_y, sup_vec in zip(self.alphas, self.sup_vec_y, self.sup_vec):
                    s += alphas * sup_vec_y * self.kernel(X[i], sup_vec)
                y_predict[i] = s
            return np.sign(y_predict + self.b)


svm = SVM(rbf_kernel, 0.1)
#SVM Fitting
svm.fit(x_train, y_train)
#Prediction
prediction = svm.predict(x_test)

res=pd.DataFrame(data={'Bound': prediction.astype(int)})
res['Id']=0
for i in range(len(res)):
      res['Id'][i]=i
y_test = pd.DataFrame(data={'Id':res['Id'],'Bound': res['Bound']})
y_test['Bound'][y_test['Bound'] == -1] = 0
y_test.to_csv('Y_te.csv', index=False)