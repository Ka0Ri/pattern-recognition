import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# print("ex: 1.3.1")
# m = np.array([0, 1])
# S = np.identity(2)
# pg1 = multivariate_normal.pdf(np.array([0.2, 1.3]), m, S)
# pg2 = multivariate_normal.pdf(np.array([2.2, -1.3]), m, S)
# print("probability of x1: ", pg1)
# print("probability of x2: ", pg2)

print("ex: 1.3.2")
# m1 = np.array([1, 1])
# m2 = np.array([3, 3])
# S = np.identity(2)
# p1 = 1/2*multivariate_normal.pdf([1.8, 1.8], m1, S)
# p2 = 1/2*multivariate_normal.pdf([1.8, 1.8], m2, S)
# print("p1: ", p1)
# print("p2: ", p2)
# print("x belongs to the first class")

print("excercise 1.3.1")
# print("p(w1) = 1/6, p(w2) = 5/6")
# p1 = 1/6*multivariate_normal.pdf([1.8, 1.8], m1, S)
# p2 = 5/6*multivariate_normal.pdf([1.8, 1.8], m2, S)
# print("p1: ", p1)
# print("p2: ", p2)
# print("x belongs to the second class")
# print("p(w1) = 5/6, p(w2) = 1/6")
# p1 = 5/6*multivariate_normal.pdf([1.8, 1.8], m1, S)
# p2 = 1/6*multivariate_normal.pdf([1.8, 1.8], m2, S)
# print("p1: ", p1)
# print("p2: ", p2)
# print("x belongs to the first class")

print("ex 1.3.3")
# mean = [0, 0]
# x, y = np.random.multivariate_normal(mean, [[1,0],[0,1]], 500).T
# plt.plot(x, y, 'o')
# plt.show()

# x, y = np.random.multivariate_normal(mean, [[0.2,0],[0,0.2]], 500).T
# plt.plot(x, y, 'o')
# plt.show()

# x, y = np.random.multivariate_normal(mean, [[2,0],[0,2]], 500).T
# plt.plot(x, y, 'o')
# plt.show()

# x, y = np.random.multivariate_normal(mean, [[0.2,0],[0,2]], 500).T
# plt.plot(x, y, 'o')
# plt.show()

# x, y = np.random.multivariate_normal(mean, [[2,0],[0,0.2]], 500).T
# plt.plot(x, y, 'o')
# plt.show()

# x, y = np.random.multivariate_normal(mean, [[1,0.5],[0.5,1]], 500).T
# plt.plot(x, y, 'o')
# plt.show()

# x, y = np.random.multivariate_normal(mean, [[0.3,0.5],[0.5,2]], 500).T
# plt.plot(x, y, 'o')
# plt.show()

# x, y = np.random.multivariate_normal(mean, [[0.3,-0.5],[-0.5,2]], 500).T
# plt.plot(x, y, 'o')
# plt.show()