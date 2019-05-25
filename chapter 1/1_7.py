import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from auxiliary import random_MixtureGaussian, pazen_window, generate_sample

print("ex 1.7.1")
# N = 10000
# m = np.array([[0],
#             [2]])
# S = np.array([[[0.2]],
#             [[0.2]]])
# P = np.array([1/3, 2/3])
# X = random_MixtureGaussian(N, P, m, S)
# x = np.linspace(-5, 5, 1000)
# y = (1/3)*multivariate_normal(0, 0.2).pdf(x) + (2/3)*multivariate_normal(2, 0.2).pdf(x)
# # plt.plot(x, y, 'b-')
# # plt.show()
# p = pazen_window(np.reshape(x, (1000, 1)), X, 0.1)
# plt.plot(x, p, 'b-')
# plt.show()

print("exercise 1.7.2")
# N = 1000
# m = np.array([[0, 0],
#             [0, 2]])
# S = np.array([[[0.2, 0], [0, 0.2]],
#             [[0.2, 0], [0, 0.2]]])
# P = np.array([1/3, 2/3])
# data = random_MixtureGaussian(N, P, m, S)

# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x, y)
# pos = np.dstack((X, Y))

# Z_true = P[0]*multivariate_normal(m[0], S[0]).pdf(pos) + P[1]*multivariate_normal(m[1], S[1]).pdf(pos)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(X, Y, Z_true)
# # plt.show()
# Z = pazen_window(np.reshape(pos, (100*100, 2)), data, 0.1)
# ax.plot_surface(X, Y, np.reshape(Z, (100, 100)))
# plt.show()


# print("exercise 1.7.3")
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
# prob1 = P[0]*pazen_window(X_test, x1, 1)
# prob2 = P[1]*pazen_window(X_test, x2, 1)
# prob3 = P[2]*pazen_window(X_test, x3, 1)

# prob = np.stack([prob1, prob2, prob3])
# predict = np.argmax(prob, axis=0)
# acc = np.sum(predict == Y_test)/(Y_test.shape[0])
# print("Bayes: ", acc)