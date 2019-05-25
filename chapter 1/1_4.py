import numpy as np
from scipy.stats import multivariate_normal
from auxiliary import Bayes_Normal_classifier, generate_sample, Distance_classifier

print("ex 1.4.1")
print("Euclidian distance")
# x = np.array([0.1, 0.5, 0.1])
# m1 = np.array([0, 0, 0])
# m2 = np.array([0.5, 0.5, 0.5])
# d1 = np.linalg.norm(x-m1)
# d2 = np.linalg.norm(x-m2)
# print(d1, d2)
# print("Mahalanbonis distance")
# x = np.array([0.1, 0.5, 0.1])
# m1 = np.array([0, 0, 0])
# m2 = np.array([0.5, 0.5, 0.5])
# S = np.array([[0.8, 0.01, 0.01], [0.01, 0.2, 0.01], [0.01, 0.01, 0.2]])

# d1 = np.sqrt(np.dot(np.dot((x-m1), np.linalg.inv(S)),(x-m1).T))
# d2 = np.sqrt(np.dot(np.dot((x-m2), np.linalg.inv(S)),(x-m2).T))
# print(d1, d2)

print("ex 1.4.2")
# x, y = np.random.multivariate_normal([2, -2], [[0.9, 0.2], [0.2, 0.3]], 50).T
# mean = np.mean([x, y], axis=1)
# cov = np.cov(x, y)
# print(mean)
# print(cov)
# x, y = np.random.multivariate_normal([2, -2], [[0.9, 0.2], [0.2, 0.3]], 500).T
# mean = np.mean([x, y], axis=1)
# cov = np.cov(x, y)
# print(mean)
# print(cov)
# x, y = np.random.multivariate_normal([2, -2], [[0.9, 0.2], [0.2, 0.3]], 5000).T
# mean = np.mean([x, y], axis=1)
# cov = np.cov(x, y)
# print(mean)
# print(cov)

print("ex 1.4.3")
# N = 1000
# m1 = np.array([[0, 0, 0],
#         [1, 2, 2],
#         [3, 3, 4]])
# S = np.array([0.8*np.identity(3), 0.8*np.identity(3), 0.8*np.identity(3)])
# P = np.array([1/3, 1/3, 1/3])

# X_train, Y_train = generate_sample(N, P, m1, S)
# X_test, Y_test = generate_sample(N, P, m1, S)
# x1 = X_train[np.where(Y_train == 0)]
# x2 = X_train[np.where(Y_train == 1)]
# x3 = X_train[np.where(Y_train == 2)]

# em1 = np.mean(x1, axis=0)
# em2 = np.mean(x2, axis=0)
# em3 = np.mean(x3, axis=0)
# ecov1 = np.cov(x1.T)
# ecov2 = np.cov(x2.T)
# ecov3 = np.cov(x3.T)
# print(em1)
# print(ecov1)
# print(em2)
# print(ecov2)
# print(em3)
# print(ecov3)

# print("step 2")
# acc, pre = Distance_classifier(3, X_test, Y_test, X_train, Y_train)
# print("Euclid: ", acc)

# print("step 3")
# acc, pre = Distance_classifier(3, X_test, Y_test, X_train, Y_train, mode="Maha")
# print("Maha: ", acc)

# print("step 4")
# acc, _ = Bayes_Normal_classifier(X_test, Y_test, P, np.stack([em1, em2, em3]), np.stack([ecov1, ecov2, ecov3]))
# print("Bayes: ", acc)

print("exersice 1.4.2")
# N = 1000
# m1 = np.array([[0, 0, 0],
#         [1, 2, 2],
#         [3, 3, 4]])
# S1 = np.array([[0.8, 0.2, 0.1], [0.2, 0.8, 0.2], [0.1, 0.2, 0.8]])
# S = np.array([S1, S1, S1])
# P = np.array([1/3, 1/3, 1/3])

# X_train, Y_train = generate_sample(N, P, m1, S)
# X_test, Y_test = generate_sample(N, P, m1, S)
# x1 = X_train[np.where(Y_train == 0)]
# x2 = X_train[np.where(Y_train == 1)]
# x3 = X_train[np.where(Y_train == 2)]

# em1 = np.mean(x1, axis=0)
# em2 = np.mean(x2, axis=0)
# em3 = np.mean(x3, axis=0)
# ecov1 = np.cov(x1.T)
# ecov2 = np.cov(x2.T)
# ecov3 = np.cov(x3.T)
# print(em1)
# print(ecov1)
# print(em2)
# print(ecov2)
# print(em3)
# print(ecov3)

# print("step 2")
# acc, pre = Distance_classifier(3, X_test, Y_test, X_train, Y_train)
# print("Euclid: ", acc)

# print("step 3")
# acc, pre = Distance_classifier(3, X_test, Y_test, X_train, Y_train, mode="Maha")
# print("Maha: ", acc)

# print("step 4")
# acc, _ = Bayes_Normal_classifier(X_test, Y_test, P, np.stack([em1, em2, em3]), np.stack([ecov1, ecov2, ecov3]))
# print("Bayes: ", acc)

print("exersice 1.4.3")
# N = 1000
# m1 = np.array([[0, 0, 0],
#         [1, 2, 2],
#         [3, 3, 4]])
# S = np.array([0.8*np.identity(3), 0.8*np.identity(3), 0.8*np.identity(3)])

# P = np.array([1/2, 1/4, 1/4])

# X_train, Y_train = generate_sample(N, P, m1, S)
# X_test, Y_test = generate_sample(N, P, m1, S)
# x1 = X_train[np.where(Y_train == 0)]
# x2 = X_train[np.where(Y_train == 1)]
# x3 = X_train[np.where(Y_train == 2)]

# em1 = np.mean(x1, axis=0)
# em2 = np.mean(x2, axis=0)
# em3 = np.mean(x3, axis=0)
# ecov1 = np.cov(x1.T)
# ecov2 = np.cov(x2.T)
# ecov3 = np.cov(x3.T)
# print(em1)
# print(ecov1)
# print(em2)
# print(ecov2)
# print(em3)
# print(ecov3)

# print("step 2")
# acc, pre = Distance_classifier(3, X_test, Y_test, X_train, Y_train)
# print("Euclid: ", acc)

# print("step 3")
# acc, pre = Distance_classifier(3, X_test, Y_test, X_train, Y_train, mode="Maha")
# print("Maha: ", acc)

# print("step 4")
# acc, _ = Bayes_Normal_classifier(X_test, Y_test, P, np.stack([em1, em2, em3]), np.stack([ecov1, ecov2, ecov3]))
# print("Bayes: ", acc)

print("exersice 1.4.4")
# N = 1000
# m1 = np.array([[0, 0, 0],
#         [1, 2, 2],
#         [3, 3, 4]])
# S = np.array([[[0.8, 0.2, 0.1], [0.2, 0.8, 0.2], [0.1, 0.2, 0.8]],
#             [[0.6, 0.01, 0.01], [0.01, 0.8, 0.01], [0.01, 0.01, 0.6]],
#             [[0.6, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.6]]])

# P = np.array([1/3, 1/3, 1/3])

# X_train, Y_train = generate_sample(N, P, m1, S)
# X_test, Y_test = generate_sample(N, P, m1, S)
# x1 = X_train[np.where(Y_train == 0)]
# x2 = X_train[np.where(Y_train == 1)]
# x3 = X_train[np.where(Y_train == 2)]

# em1 = np.mean(x1, axis=0)
# em2 = np.mean(x2, axis=0)
# em3 = np.mean(x3, axis=0)
# ecov1 = np.cov(x1.T)
# ecov2 = np.cov(x2.T)
# ecov3 = np.cov(x3.T)
# print(em1)
# print(ecov1)
# print(em2)
# print(ecov2)
# print(em3)
# print(ecov3)

# print("step 2")
# acc, pre = Distance_classifier(3, X_test, Y_test, X_train, Y_train)
# print("Euclid: ", acc)

# print("step 3")
# acc, pre = Distance_classifier(3, X_test, Y_test, X_train, Y_train, mode="Maha")
# print("Maha: ", acc)

# print("step 4")
# acc, _ = Bayes_Normal_classifier(X_test, Y_test, P, np.stack([em1, em2, em3]), np.stack([ecov1, ecov2, ecov3]))
# print("Bayes: ", acc)