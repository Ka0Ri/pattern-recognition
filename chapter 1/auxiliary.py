import numpy as np
from scipy.stats import multivariate_normal


def Distance_classifier(n_class, X_test, Y_test, X_train, Y_train, mode="Euclid"):
    n = X_test.shape[0]
    dist = np.zeros([1, n])
    for i in range(n_class):
        x = X_train[np.where(Y_train == i)]
        em = np.mean(x, axis=0)
        ecov = np.cov(x.T)
        if(mode == "Euclid"):
            d = np.linalg.norm(X_test-em, axis=1)
        if(mode == "Maha"):
            d = (np.dot(np.dot((X_test-em), np.linalg.inv(ecov)),(X_test-em).T))
            d = np.sqrt(d.diagonal())
        dist = np.concatenate([dist, [d]])
    dist = dist[1:]
    predict = np.argmin(dist, axis=0)
    acc = np.sum(predict == Y_test)/(Y_test.shape[0])
    return acc, predict

def Bayes_Normal_classifier(X_test, Y_test, Priors, mus, sigmas):
    k = Priors.shape[0]
    n = X_test.shape[0]
    P = np.zeros([1, n])
    for i in range(k):
        prob = Priors[i]*multivariate_normal.pdf(X_test, mus[i], sigmas[i])
        P = np.concatenate([P, np.array([prob])])
    P = P[1:]

    predict = np.argmax(P, axis=0)
    acc = np.sum(predict == Y_test)/(n)
    return acc, predict

def generate_sample(N, P, mus, sigmas):
    m = mus.shape[1]
    k = P.shape[0]
    ns = (N*P).astype(int)
    X = np.zeros((1,m))
    Y = np.array([0])
    for i in range(k):
        x = np.random.multivariate_normal(mus[i], sigmas[i], ns[i])
        y = i*np.ones([ns[i]])
        X = np.concatenate([X, x])
        Y = np.concatenate([Y, y])
    return X[1:], Y[1:]

def random_MixtureGaussian(N, P, m, S):
    d = m.shape[1]
    c = m.shape[0]
    z = np.random.choice(np.arange(0,c), N, p=P)
    X = np.ones([1,d])
    for i in range(0,c):
        n = np.sum(z == np.array([i]*N))
        x = np.random.multivariate_normal(m[i], S[i], n)
        X = np.concatenate([X, x])
    X = X[1:]
    return X

def EM_GMM(X, pz, mus, sigmas, tol=0.01, max_iter=100):
    n, p = X.shape[:]
    k = pz.shape[0]

    ll_old = 0
    for i in range(max_iter):
        # E-step
        T = np.zeros((k, n))
        for j in range(k):
            for i in range(n):
                T[j, i] = pz[j] * multivariate_normal(mus[j], sigmas[j]).pdf(X[i])
        T /= T.sum(0)

        # M-step
        #update mixture propability
        pz = np.zeros(k)
        for j in range(k):
            for i in range(n):
                pz[j] += T[j, i]
        pz /= n

        #update means
        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += T[j, i] * X[i]
            mus[j] /= T[j, :].sum()

        #update covariance matrixs
        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(X[i]- mus[j], (2,1))
                sigmas[j] += T[j, i] * np.dot(ys, ys.T)
            sigmas[j] /= T[j,:].sum()
        # update complete log likelihoood
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pz[j] * multivariate_normal(mus[j], sigmas[j]).pdf(X[i])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_old, pz, mus, sigmas

def pazen_window(x, data, h):
    m = x.shape[0]
    n = data.shape[0]
    l = data.shape[1]
    p = np.zeros(m)
    for j in range(m):
        for i in range(n):
            dot = (np.linalg.norm(x[j] - data[i]))**2
            p[j] = p[j] + np.exp(-dot/(2*h*h))
    p = (p/(n*(pow(2*np.pi,l/2))*pow(h,l)))
    return p

def kNN_estimator(x, data, k):
    m = x.shape[0]
    n = data.shape[0]
    l = data.shape[1]
    p = np.zeros(m)

    if(l % 2 == 0):
        t = l//2
        V = (np.pi**t)/np.math.factorial(t)
    else:
        t = (l-1)//2
        V = (2*np.math.factorial(t))*pow(4*np.pi, t)/np.math.factorial(2*t + 1)
    if(l == 1):
        V = 2
    for j in range(m):
        distances = np.linalg.norm(data - x[j], axis=1)
        distances = np.sort(distances)
        dist = distances[k-1]
        
        p[j] = k/(n*V*(pow(dist, l)))
       
    return p

def kNN_classifier(X_test, Y_test, X_train, Y_train, k):
    m = X_test.shape[0]
    n = X_train.shape[0]
    predict = np.zeros(m)
    for j in range(m):
        distances = np.linalg.norm(X_train - X_test[j], axis=1)
        sorted_index = np.argsort(distances)
        k_class = Y_train[sorted_index[:k]]
        unique, counts = np.unique(k_class, return_counts=True)
        predict[j] = unique[np.argmax(counts)]
    
    acc = np.sum(predict == Y_test)/m
    return acc, predict