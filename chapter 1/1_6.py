import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from auxiliary import random_MixtureGaussian, EM_GMM



print("ex 1.6.1")

# m = np.array([[1, 1],
#             [3,3],
#             [2,6]])
# S = np.array([[[0.1, 0], [0, 0.1]],
#             [[0.2, 0], [0, 0.2]],
#             [[0.3, 0], [0, 0.3]]])
# P = np.array([0.4, 0.4, 0.2])
# N = 500
# X = random_MixtureGaussian(N, P, m, S)
# plt.axis('equal')
# plt.plot(X[:,0], X[:,1], 'o')
# plt.show()

# pz = np.array([1/3, 1/3, 1/3])
# mus = np.array([[0, 2],
#                 [5, 2], 
#                 [5, 5]])
# sigmas = np.array([[[0.15, 0],[0, 0.15]],
#                     [[0.27, 0], [0, 0.27]],
#                     [[0.4, 0], [0, 0.4]]])
# log, pz, mus, sigmas = EM_GMM(X, pz, mus, sigmas)
# print(pz)
# print(mus)
# print(sigmas)

print("ex 1.6.2")
N1 = 500
m1 = np.array([[1.25, 1.25],
                [2.75, 2.75],
                [2, 6]])
S1 = np.array([[[0.1, 0], [0, 0.1]],
                [[0.2, 0], [0, 0.2]],
                [[0.3, 0], [0, 0.3]]])
P1 = np.array([0.4, 0.4, 0.2])
X1 = random_MixtureGaussian(N1, P1, m1, S1)
X1_test = random_MixtureGaussian(N1, P1, m1, S1)
N2 = 500
m2 = np.array([[1.25, 2.75],
                [2.75, 1.25],
                [4, 6]])
S2 = np.array([[[0.1, 0], [0, 0.1]],
                [[0.2, 0], [0, 0.2]],
                [[0.3, 0], [0, 0.3]]])
P2 = np.array([0.2, 0.3, 0.5])
X2 = random_MixtureGaussian(N2, P2, m2, S2)
X2_test = random_MixtureGaussian(N2, P2, m2, S2)

X_test = np.concatenate([X1_test, X2_test])
Y_test = np.concatenate([np.array([0]*N1), np.array([1]*N2)])
# # plt.axis('equal')
# # plt.plot(X1[:,0], X1[:,1], 'bo')
# # plt.plot(X2[:,0], X2[:,1], 'r+')
# # plt.show()

# print("step 1")
# pz = np.array([1/3, 1/3, 1/3])
# mus = np.array([[0, 2],
#                 [5, 2], 
#                 [5, 5]])
# sigmas = np.array([[[0.15, 0],[0, 0.15]],
#                     [[0.27, 0], [0, 0.27]],
#                     [[0.4, 0], [0, 0.4]]])
# log1, pz1, mus1, sigmas1 = EM_GMM(X1, pz, mus, sigmas)
# print(pz1)
# print(mus1)
# print(sigmas1)
# log2, pz2, mus2, sigmas2 = EM_GMM(X2, pz, mus, sigmas)
# print(pz2)
# print(mus2)
# print(sigmas2)

# bayes_p1 = 0
# for i in range(3):
#     bayes_p1 += pz1[i]*multivariate_normal(mus1[i], sigmas1[i]).pdf(X_test)
# bayes_p2 = 0
# for i in range(3):
#     bayes_p2 += pz2[i]*multivariate_normal(mus2[i], sigmas2[i]).pdf(X_test)

# bayes_p1 = 0.5*bayes_p1
# bayes_p2 = 0.5*bayes_p2
# prob = np.stack([bayes_p1, bayes_p2])
# predict = np.argmax(prob, axis=0)
# acc = np.sum(predict == Y_test)/(Y_test.shape[0])
# print("Bayes: ", acc)

# print("step 1")
# pz = np.array([1/4, 1/4, 1/4, 1/4])
# mus = np.array([[0, 2],
#                 [5, 2], 
#                 [5, 5],
#                 [3, 4]])
# sigmas = np.array([[[0.15, 0],[0, 0.15]],
#                     [[0.27, 0], [0, 0.27]],
#                     [[0.4, 0], [0, 0.4]],
#                     [[0.2, 0], [0, 0.2]]])
# log1, pz1, mus1, sigmas1 = EM_GMM(X1, pz, mus, sigmas)
# print(pz1)
# print(mus1)
# print(sigmas1)
# mus = np.array([[1, 2],
#                 [3.2, 1.5], 
#                 [1, 4],
#                 [4, 2]])
# sigmas = np.array([[[0.15, 0],[0, 0.15]],
#                     [[0.08, 0], [0, 0.08]],
#                     [[0.27, 0], [0, 0.27]],
#                     [[0.05, 0], [0, 0.05]]])
# log2, pz2, mus2, sigmas2 = EM_GMM(X2, pz, mus, sigmas)
# print(pz2)
# print(mus2)
# print(sigmas2)

# bayes_p1 = 0
# for i in range(pz1.shape[0]):
#     bayes_p1 += pz1[i]*multivariate_normal(mus1[i], sigmas1[i]).pdf(X_test)
# bayes_p2 = 0
# for i in range(pz2.shape[0]):
#     bayes_p2 += pz2[i]*multivariate_normal(mus2[i], sigmas2[i]).pdf(X_test)

# bayes_p1 = 0.5*bayes_p1
# bayes_p2 = 0.5*bayes_p2
# prob = np.stack([bayes_p1, bayes_p2])
# predict = np.argmax(prob, axis=0)
# acc = np.sum(predict == Y_test)/(Y_test.shape[0])
# print("Bayes: ", acc)
        
# print("step 1")
# pz = np.array([1])
# mus = np.array([[2, 2]])
# sigmas = np.array([[[0.4, 0], [0, 0.4]]])
# log1, pz1, mus1, sigmas1 = EM_GMM(X1, pz, mus, sigmas)
# print(pz1)
# print(mus1)
# print(sigmas1)
# mus = np.array([[1, 2]])
# sigmas = np.array([[[0.15, 0], [0, 0.15]]])
# log2, pz2, mus2, sigmas2 = EM_GMM(X2, pz, mus, sigmas)
# print(pz2)
# print(mus2)
# print(sigmas2)

# bayes_p1 = 0
# for i in range(1):
#     bayes_p1 += pz1[i]*multivariate_normal(mus1[i], sigmas1[i]).pdf(X_test)
# bayes_p2 = 0
# for i in range(1):
#     bayes_p2 += pz2[i]*multivariate_normal(mus2[i], sigmas2[i]).pdf(X_test)

# bayes_p1 = 0.5*bayes_p1
# bayes_p2 = 0.5*bayes_p2
# prob = np.stack([bayes_p1, bayes_p2])
# predict = np.argmax(prob, axis=0)
# acc = np.sum(predict == Y_test)/(Y_test.shape[0])
# print("Bayes: ", acc)

print("step 1")
pz = np.array([1/2, 1/2])
mus = np.array([[0, 2],
                [5, 2]])
sigmas = np.array([[[0.15, 0], [0, 0.15]],
                    [[0.27, 0], [0, 0.27]]])
log1, pz1, mus1, sigmas1 = EM_GMM(X1, pz, mus, sigmas)
print(pz1)
print(mus1)
print(sigmas1)
pz = np.array([1])
mus = np.array([[1, 2]])
sigmas = np.array([[[0.15, 0], [0, 0.15]]])
log2, pz2, mus2, sigmas2 = EM_GMM(X2, pz, mus, sigmas)
print(pz2)
print(mus2)
print(sigmas2)

bayes_p1 = 0
for i in range(pz1.shape[0]):
    bayes_p1 += pz1[i]*multivariate_normal(mus1[i], sigmas1[i]).pdf(X_test)
bayes_p2 = 0
for i in range(pz2.shape[0]):
    bayes_p2 += pz2[i]*multivariate_normal(mus2[i], sigmas2[i]).pdf(X_test)

bayes_p1 = 0.5*bayes_p1
bayes_p2 = 0.5*bayes_p2
prob = np.stack([bayes_p1, bayes_p2])
predict = np.argmax(prob, axis=0)
acc = np.sum(predict == Y_test)/(Y_test.shape[0])
print("Bayes: ", acc)