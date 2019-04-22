# Clustering

## General

**Clustering Problem**: given samples $\{{\bf x}_i\}_{i=1}^N$, assign points to *disjoint* subsets called *clusters* so that points in the same cluster are more similar to each other than points in different clusters.

- Clustering is finding a map $C: [1:N]\to [1:K]$ with $K$ the number of clusters.

## K-means

### Theory

K-means clustering aims to find $C​$ that minimizes $W(C)​$ with
$$
W(C)=\frac{1}{2}\sum_{k=1}^K\sum_{i:C(i)=k}\sum_{j:C(j)=k}\|{\bf x}_i-{\bf x}_j\|_2^2
$$
The number of possible clusters is given by *Stirling's numbers of the second kind*:
$$
S(N,K)=\frac{1}{K!}\sum_{k=1}^K(-1)^{K-k}\left(\begin{matrix}
K\\k
\end{matrix}\right)k^N
$$
and there is no efficient search strategy for this space.

------

Instead, we switch to finding a sub-optimal k-means clustering $C^*​$:
$$
C^*= \underset{C}{\operatorname{argmin}}\sum_{k=1}^KN_k\sum_{i:C(i)=k}\|{\bf x}_i-{\pmb \mu}_k\|_2^2
$$

where ${\pmb \mu}_k=\frac{1}{N_k}\sum_{i:C(i)=k}{\bf x}_i$ and $N_k = |\{ i:C(i)=k \}|$ which is the number of sample points fall in class $k$.

- For a fixed $C$ we have
  $$
  \hat{\pmb \mu}_k=\underset{{\pmb \mu}_k}{\operatorname{argmin}} \sum_{i:C(i)=k} \|{\bf x}_i-{\pmb \mu}_k\|_2^2
  $$
  and solving (4) we get
  $$
  \hat{\pmb \mu}_k = \frac{1}{N_k} \sum_{i:C(i)=k} {\bf x}_i
  $$

- For a fixed $\{{\pmb \mu}_k\}_{k=1}^K​$ we have
  $$
  \hat{C}= \underset{C}{\operatorname{argmin}}\sum_{k=1}^KN_k\sum_{i:C(i)=k}\|{\bf x}_i-{\pmb \mu}_k\|_2^2
  $$
  and solving (6) we get
  $$
  C(i)=\underset{k}{\operatorname{argmin}}\|{\bf x}_i-{\pmb \mu}_k\|_2^2
  $$

The visualization of the procedure is as below:

![k-means visualization](<https://raw.githubusercontent.com/Yinghao-Li/Yinghao-Li.github.io/master/img/post-Clustering/post-Clustering-K-means.PNG>)

### Notes

- Algorithm typically initialized with ${\pmb \mu}_k$ as random points in dataset;
- Several random initialization to avoid local minima;
- Cluster boundaries are parts of hyperplane;
- K-means **fails** if the clusters are non-convex;
- Geometry changes if we change the $\ell_2$ norm.

### Realization in Python

Function definition:

```python
import numpy as np

def KMeans(X, k, iter_time=None, epsilon=1e-6):
    N, d = X.shape
    ini_idx = (np.random.rand(k, 1) * N).astype(np.int)
    means = X[ini_idx, :].reshape([k, d])
    pre_means = np.zeros(means.shape)
    
    t = 0
    while np.sum((means - pre_means) ** 2) > epsilon:
        pre_means = means.copy()
        distances = np.zeros([N, k])
        for i in range(k):
            distances[:, i] = np.sum((X - means[i, :]) ** 2, axis=1)
        classes = np.argmin(distances, axis=1)
        means = np.zeros(means.shape)
        for i in range(N):
            means[classes[i], :] += X[i, :]
        for i in range(k):
            means[i, :] /= np.sum(classes == i)
        t += 1
        if t==iter_time:
            break
    return means, classes
```

Testing code:

```python
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

means, classes = KMeans(X, 4)

plt.scatter(X[:, 0], X[:, 1], c=classes, s=50, cmap='viridis')
plt.scatter(means[:, 0], means[:, 1], c='black', s=200, alpha=0.5);
```

## Gaussian Mixture Model

