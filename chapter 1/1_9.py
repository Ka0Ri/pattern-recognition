import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from auxiliary import generate_sample, Bayes_Normal_classifier

# print("ex: 1.9.1")

# N1 = 1000
# m1 = np.array([[0, 0, 0, 0, 0],
#                 [1, 1, 1, 1, 1]])
# S1 = np.array([[[0.8, 0.2, 0.1, 0.05, 0.01], [0.2, 0.7, 0.1, 0.03, 0.02], [0.1, 0.1, 0.8, 0.02, 0.01], [0.05, 0.03, 0.02, 0.9, 0.01], [0.01, 0.02, 0.01, 0.01, 0.8]],
#                 [[0.9, 0.1, 0.05, 0.02, 0.01], [0.1, 0.8, 0.1, 0.02, 0.02], [0.05, 0.1, 0.7, 0.02, 0.01], [0.02, 0.02, 0.02, 0.6, 0.02], [0.01, 0.02, 0.01, 0.02, 0.7]]])
# P1 = np.array([0.5, 0.5])
# X_train, Y_train = generate_sample(N1, P1, m1, S1)

# X_train1 = X_train[np.where(Y_train == 0)]
# X_train2 = X_train[np.where(Y_train == 1)]

# em1 = np.mean(X_train1, axis=0)
# em2 = np.mean(X_train2, axis=0)
# evar1 = np.var(X_train1, axis=0)
# evar2 = np.var(X_train2, axis=0)
# ecov1 = np.diag(evar1)
# ecov2 = np.diag(evar2)
# cov1 = np.cov(X_train1.T)
# cov2 = np.cov(X_train2.T)
# print(em1)
# print(em2)
# print(ecov1)
# print(ecov2)
# X_test, Y_test = generate_sample(10000, P1, m1, S1)
# acc, predict = Bayes_Normal_classifier(X_test, Y_test, P1, np.stack([em1, em2]), np.stack([cov1, cov2]))
# print("Bayes: ", acc)
