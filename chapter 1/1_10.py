import numpy as np
from auxiliary import generate_sample, kNN_classifier, Bayes_Normal_classifier
import time

print("1.10.1 ")
# m1 = np.array([[0, 0],
#                 [1, 2]])
# S1 = np.array([[[0.8, 0.2], [0.2, 0.8]],
#                 [[0.8, 0.2], [0.2, 0.8]]])
# P = np.array([0.5, 0.5])
# N1 = 1000
# N2 = 5000
# X_train, Y_train = generate_sample(N1, P, m1, S1)
# X_test, Y_test = generate_sample(N2, P, m1, S1)

# x1 = X_train[np.where(Y_train == 0)]
# x2 = X_train[np.where(Y_train == 1)]
# em1 = np.mean(x1, axis=0)
# em2 = np.mean(x2, axis=0)
# ecov1 = np.cov(x1.T)
# ecov2 = np.cov(x2.T)
# acc, _ = Bayes_Normal_classifier(X_test, Y_test, P, np.stack([em1, em2]), np.stack([ecov1, ecov2]))
# print(acc)
# acc, pred = kNN_classifier(X_test, Y_test, X_train, Y_train, 1)
# print(acc)
# acc, pred = kNN_classifier(X_test, Y_test, X_train, Y_train, 3)
# print(acc)
# acc, pred = kNN_classifier(X_test, Y_test, X_train, Y_train, 7)
# print(acc)
# acc, pred = kNN_classifier(X_test, Y_test, X_train, Y_train, 15)
# print(acc)

# N1 = 1000
# m1 = np.array([[0, 0, 0, 0, 0],
#                 [1, 1, 1, 1, 1]])
# S1 = np.array([[[0.8, 0.2, 0.1, 0.05, 0.01], [0.2, 0.7, 0.1, 0.03, 0.02], [0.1, 0.1, 0.8, 0.02, 0.01], [0.05, 0.03, 0.02, 0.9, 0.01], [0.01, 0.02, 0.01, 0.01, 0.8]],
#                 [[0.9, 0.1, 0.05, 0.02, 0.01], [0.1, 0.8, 0.1, 0.02, 0.02], [0.05, 0.1, 0.7, 0.02, 0.01], [0.02, 0.02, 0.02, 0.6, 0.02], [0.01, 0.02, 0.01, 0.02, 0.7]]])
# P1 = np.array([0.5, 0.5])
# X_train, Y_train = generate_sample(N1, P1, m1, S1)

# X_train1 = X_train[np.where(Y_train == 0)]
# X_train2 = X_train[np.where(Y_train == 1)]
# X_test, Y_test = generate_sample(10000, P1, m1, S1)

# em1 = np.mean(X_train1, axis=0)
# em2 = np.mean(X_train2, axis=0)
# cov1 = np.cov(X_train1.T)
# cov2 = np.cov(X_train2.T)
# acc, predict = Bayes_Normal_classifier(X_test, Y_test, P1, np.stack([em1, em2]), np.stack([cov1, cov2]))
# print("Bayes: ", acc)

# X_test, Y_test = generate_sample(1000, P1, m1, S1)
# t = time.time()
# acc, predict = kNN_classifier(X_test, Y_test, X_train, Y_train, 1)
# t_1 = time.time()
# print("KNN: ", acc, "Time: ", t_1 - t)

# t = time.time()
# acc, predict = kNN_classifier(X_test, Y_test, X_train, Y_train, 5)
# t_1 = time.time()
# print("KNN: ", acc, "Time: ", t_1 - t)

# t = time.time()
# acc, predict = kNN_classifier(X_test, Y_test, X_train, Y_train, 7)
# t_1 = time.time()
# print("KNN: ", acc, "Time: ", t_1 - t)

# t = time.time()
# acc, predict = kNN_classifier(X_test, Y_test, X_train, Y_train, 15)
# t_1 = time.time()
# print("KNN: ", acc, "Time: ", t_1 - t)