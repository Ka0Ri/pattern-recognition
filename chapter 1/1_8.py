import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from auxiliary import generate_sample, random_MixtureGaussian, kNN_estimator

print("ex 1.8.1")
# N = 1000
# m = np.array([[0],
#             [2]])
# S = np.array([[[0.2]],
#             [[0.2]]])
# P = np.array([1/3, 2/3])
# X = random_MixtureGaussian(N, P, m, S)

# x = np.linspace(-5, 5, 1000)
# y = 1/3*multivariate_normal(0, 0.2).pdf(x) + 2/3*multivariate_normal(2, 0.2).pdf(x)
# # plt.plot(x, y, 'b-')
# # plt.show()
# p = kNN_estimator(np.reshape(x, (1000, 1)), X, 5)
# plt.plot(x, p, 'b-')
# plt.show()

print("exercise 1.8.2")

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

# print("step 4")
# prob1 = P[0]*kNN_estimator(X_test, x1, 5)
# prob2 = P[1]*kNN_estimator(X_test, x2, 5)
# prob3 = P[2]*kNN_estimator(X_test, x3, 5)

# prob = np.stack([prob1, prob2, prob3])
# predict = np.argmax(prob, axis=0)
# acc = np.sum(predict == Y_test)/(Y_test.shape[0])
# print("Bayes: ", acc)