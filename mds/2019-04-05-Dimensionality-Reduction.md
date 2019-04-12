# Dimensionality Reduction

[TOC]

## Overview

Feature vectors in dataset $\mathcal{D}$ are ${\bf x}_i \in \mathbb{R}^d$, with dimension $d$ potentially large. Sometimes redundant dimensions would be introduced and in this case, the efficiency of training process would be jeopardized and overfitting is more likely to happen. So we want to reduce the dimension of dataset as well as preserve the information in the dataset as much as possible.

Dimensionality reduction aims to **transform** inputs to new variables ${\bf x}_i’ \in \mathbb{R}^k$ where $k \ll d$.

- Needs to minimize **information loss** to maintain performance;
- Often improves computational efficiency;
- Helps prevent overfitting, especially when $N \ll d​$.

## Types of Reduction Techniques

- **Information Loss**: The way information loss is measured;
- **Supervised or Unsupervised**: If the labels $y_i$ are used in the process;
- **Linearity**: If the map $\bf{x}\to\bf{x}'$ is linear;
- **Selection or Extraction**: If the features are selected from the original model or extracted and formed into a new model.

$$
\bf x_{\text{selected}}'=\left[x_1\ x_6\ x_{32}\right]^{\sf T} \qquad x_{extracted}'=\left[ \phi_1(x)\ \phi_2(x)\ \phi_3(x)\right]^{\sf T}
$$

## Filtering

> Feature selection by elimination **irrelevant** features.

object is to reduce computational complexity, regularize and retain interpretability.

Feature ranked by **order of importance** and best $k$ features are retained.

- "Importance" related to the ability to **predict** $y_i$ in supervised learning;
- Advantage: usually fast;
- Disadvantage: "$k$ best features" not always related to "best $k$ features".

![post-DR-select-extract](https://raw.githubusercontent.com/Yinghao-Li/Yinghao-Li.github.io/master/img/post-DR/post-DR-select-extract.PNG)

It is clear that in the left figure, $x_2$ should be selected and in the right one $x_1$ should be selected.

### Ranking in Classification

Many possible choices to assign a rank $r(j)$ to feature $j$:

- Misclassification rate:

$$
r(j) \triangleq \frac{1}{N}\sum_{i=1}^N\mathbb{i}\left\{y_i\neq h(x_{i,j})\right\}
$$

​	for some classifier $h$;

- Two-sample t-test statistics:

$$
r(j)\triangleq \frac{\bar{x}_j^{(+)} - \bar{x}_j^{(-)}}{s/\sqrt{n}}
$$

​	where $\bar{x}_j^{(\pm)}$ are class means for feature $j$; $s$ is the polled sample standard deviation.

- Margin:

$$
r(j)\triangleq \min_{k,\ell}\left|x_j^{\mathbb{I}(y_k=1)}-x_j^{\mathbb{I}(y_l=-1)}\right|
$$

### Ranking in Regression

- Correlation coefficient:

$$
r(j)\triangleq |\rho(j)| \text{ with } \rho(j)\triangleq\frac{{\rm Cov}(x_j,y)}{\sqrt{{\rm Var}(x_j){\rm Var}(y)}}
$$

- Mutual information:

$$
r(j)\triangleq I(X;Y) \text{ with } I(X;Y)=\sum_{x,y}p(x,y){\rm log}\frac{p(x,y)}{p(x)p(y)}
$$

### Avoiding Redundant Features

Ranking is legitimate but can lead to selection of **redundant features**.

One solution is **incremental maximization**:

> E.g. Incremental maximization with mutual information:
>
> - Assume features $x_{j_1}, \cdots, x_{j_{k-1}}$ are already selected;
> - Then we select $k^{\text{th}}$ feature that maximizes:
>
> $$
> I(x_{j_k};y)-\beta\sum_{i=1}^{k-1}I(x_{j_k};x_{j_i})
> $$

### Drawback

The biggest drawback to filtering is inability to capture **correlations** between features.

*Solutions*:

1. **Wrapper methods** based on measure of performance for subset of features.
   - Advantage: capture interactions;
   - Disadvantage: can be very slow;
   - Examples: forward selection, backward elimination.

2. **Embedded methods** based on joint feature selection and model fitting
   - Example: LASSO.

## Principle Component Analysis

> Principle Component Analysis consists in solving the problem:
> $$
> \underset{{\pmb \mu},{\bf A},{\pmb \theta}_i}{\operatorname{argmin}} = \sum_{i=1}^N\|{\bf x}_i-{\pmb \mu}-{\bf A}{\pmb \theta}_i\|_2^2
> $$
> where $\bf A$ has **orthonormal columns**, which means ${\bf A}^{\sf T}{\bf A}$ is identity matrix $\bf I$.

Assume $\pmb \mu​$ and $\bf A​$ are fixed, then,
$$
{\pmb \theta}_i = {\bf A}^{\sf T}({\bf x}_i-{\pmb \mu})\\
$$
 Assume $\bf A​$ is fixed and we calculated ${\pmb \theta}_i​$, then,
$$
{\pmb \mu} = \frac{1}{N}\sum_{i=1}^N{\bf x}_i
$$
The hard part of this calculation is finding out what $\bf A$ is.

### Solving PCA

>One possible choice of $\bf A$ is
>$$
>{\bf A}=\left[ {\bf u}_1, \dots, {\bf u}_k \right] \in \mathbb{R}^{d \times k}
>$$
>where ${\bf u}_i$ are the **eigenvectors** corresponding to the $k^{\text{th}}$ largest eigenvalues of ${\bf S} \triangleq \sum_{i=1}^N{\bf x}_i{\bf x}_i^{\sf T}$. $\bf S$ is similar to the unstandardized **covariance matrix** of the dataset.
>
>- Recall that covariance matrix has the form $\Sigma = \mathbb{E}\left[ ({\bf X}-\mathbb{E}[{\bf X}])({\bf X}-\mathbb{E}[{\bf X}])^{\sf T}\right]​$.

### Proof

#### Introduce of S

Assume we already know ${\pmb \theta}_i​$ and $\hat{\pmb \mu}​$ from (9) and (10), the problem becomes solving:
$$
\underset{{\bf A}}{\operatorname{argmin}} \sum_{i=1}^N\|{\bf x}_i-\hat{\pmb \mu}-{\bf A}{\bf A}^{\sf T}({\bf x}_i-\hat{\pmb \mu})\|_2^2
$$
s.t. ${\bf A}^{\sf T}{\bf A}={\bf I}​$.

Without loss of generality, assume $\hat{\pmb \mu}=0​$. Then we want to solve:
$$
\begin{aligned}
	& \underset{{\bf A}}{\operatorname{argmin}} \sum_{i=1}^N \|({\bf I}-{\bf A}{\bf A}^{\sf T}){\bf x}_i\|_2^2 \\
	=\ & \underset{{\bf A}}{\operatorname{argmin}} \sum_{i=1}^N {\bf x}_i^{\sf T}({\bf I}-{\bf A}{\bf A}^{\sf T}){\bf x}_i \\
	=\ & \underset{{\bf A}}{\operatorname{argmin}} \sum_{i=1}^N {\bf x}_i^{\sf T}{\bf x}_i - {\bf x}_i^{\sf T}{\bf A}{\bf A}^{\sf T}{\bf x}_i \\
	=\ & \underset{{\bf A}}{\operatorname{argmax}} \sum_{i=1}^N {\bf x}_i^{\sf T}{\bf A}{\bf A}^{\sf T}{\bf x}_i \\
	=\ & \underset{{\bf A}}{\operatorname{argmax}} \sum_{i=1}^N {\rm tr}\left( {\bf A}^{\sf T}{\bf x}_i {\bf x}_i^{\sf T}{\bf A} \right) \\
	=\ & \underset{{\bf A}}{\operatorname{argmax}} {\rm tr} \left( {\bf A}^{\sf T} \left( \sum_{i=1}^N {\bf x}_i {\bf x}_i^{\sf T} \right){\bf A} \right) \\
	=\ & \underset{{\bf A}}{\operatorname{argmax}} {\rm tr} \left( {\bf A}^{\sf T} {\bf S}{\bf A} \right)
\end{aligned}
$$

- Recall that $\|{\bf v}\|_2^2={\rm tr}({\bf v}{\bf v}^{\sf T})$ where ${\rm tr}(\bf A)$ is the trace of $\bf A$.

#### Introduce linear program

It is easy to see from definition that $\bf S​$ is positive semi-definite. Apply Eigen decomposition on $\bf S​$: ${\bf S}={\bf U}{\pmb \Lambda}{\bf U}^{\sf T}​$ where ${\bf U}\in \mathbb{R}^{d\times d}​$ is an orthonormal matrix and ${\pmb \Lambda}={\rm diag}(\lambda_1, \lambda_2, \dots, \lambda_d)​$ ,$\lambda_1\geq \lambda_2 \geq \dots \geq \lambda_d \geq 0 ​$.

Plugging it into the trace, we have:
$$
{\rm tr} \left( {\bf A}^{\sf T} {\bf S}{\bf A} \right) = {\rm tr} \left( {\bf A}^{\sf T} {\bf U}{\pmb \Lambda}{\bf U}^{\sf T}{\bf A} \right) = {\rm tr} \left( {\bf W}^{\sf T}{\pmb \Lambda} {\bf W} \right) \text{ with } {\bf W}\triangleq {\bf U}^{\sf T}{\bf A} \in \mathbb{R}^{d\times k}
$$
Note that ${\bf W}^{\sf T}{\bf W}={\bf A}^{\sf T} {\bf U}{\bf U}^{\sf T}{\bf A}={\bf I}$.

- Recall if $\bf Q$ is an orthonormal matrix then ${\bf Q}^{-1}={\bf Q}^{\sf T}$.

Then, we need to solve $\underset{{\bf W}}{\operatorname{argmax}} {\rm tr} \left( {\bf W}^{\sf T}{\pmb \Lambda} {\bf W} \right) ​$. Denote ${\bf W}\triangleq \left[ {\bf w}_1, {\bf w}_2, \dots, {\bf w}_k \right]​$ where ${\bf w}_i\in\mathbb{R}^{d\times1}​$.
$$
{\rm tr} \left( {\bf W}^{\sf T}{\pmb \Lambda} {\bf W} \right)
= \sum_{i=1}^k {\bf w}_i^{\sf T}{\pmb \Lambda}{\bf w}_i
= \sum_{i=1}^k \sum_{j=1}^d w_{ij}^2 \lambda_j
= \sum_{j=1}^d \left( \sum_{i=1}^k w_{ij}^2 \right) \lambda_j
= \sum_{j=1}^d h_j \lambda_j
$$
We know that $\lambda_j$ depends solely on our dataset $\bf X$, so we only needs to find out $h_j​$ that maximize the trace.

However, there are certain constrains on $h_j$:

1. $h_j \geq 0$;
2. $\sum_{j=1}^d h_j = \sum_{j=1}^d \sum_{i=1}^k w_{ij}^2=k$;
3. We can write ${\bf W}_1=\left[{\bf W}\ {\bf W}_0\right]\in\mathbb{R}^{d\times d}$ s.t. ${\bf W}_1^{\sf T}{\bf W}_1={\bf I}_d$ and ${\bf W}_1{\bf W}^{\sf T}_1={\bf I}_d$. In another form: $h_j=\sum_{i=1}^k w_{ij}^2 \leq \sum_{i=1}^d w_{1,ij}^2=1$. 

#### Solve Linear Program

By now, the problem have become $\underset{\{ h_j \}}{\operatorname{argmax}} \sum_{j=1}^dh_j\lambda_j​$ s.t. $0 \leq h_j \leq 1​$ and $\sum_{j=1}^dh_j=k​$. Without too much effort we can figure out that the solution is
$$
\begin{equation}
h_j=\left\{ \begin{aligned}
	&1 \text{    if  }j\in[1,k]\\
	&0 \text{    else}
\end{aligned}\right.
\end{equation}
$$
and one possible $\bf W$ is
$$
\hat{\bf W}=\left[\begin{array} &{\bf I}_k \\ {\bf 0} \end{array}\right]\in\mathbb{R}^{d\times k}
$$
Taking $\hat{\bf A}={\bf U}^{\sf T}\hat{\bf W}$, we are selecting the first $k^{\text{th}}$ columns of $\bf U$.

### Relationship to SVD

Recall that singular value decomposition takes a rectangular matrix of data with $n$ samples of $d$ features ${\bf X}\in \mathbb{R}^{n\times d}$:
$$
{\bf X}={\bf U}{\Sigma}{\bf V}^{\sf T}
$$
where ${\bf U}\in\mathbb{R}^{n\times n}$, $\Sigma\in\mathbb{R}^{n\times d}$ and ${\bf V}\in\mathbb{R}^{d\times d}$; $\bf U$ and $\bf V$ are orthonormal.

The columns of $\bf U$ are the left singular vectors; $\Sigma$ has singular values and is diagonal; ${\bf V}^{\sf T}$ has rows that are the right singular vectors.

- The eigenvectors of ${\bf X}^{\sf T}{\bf X}​$ make up the columns of $\bf V​$;
- The eigenvectors of ${\bf X}{\bf X}^{\sf T}$ make up the columns of $\bf U$;
- The singular values are the diagonal entries of the $\Sigma​$ matrix and are arranged in descending order. <font color=red>Singular values are the square roots of eigenvalues</font>.

In this case ${\bf X}={\bf U}{\Sigma}{\bf V}^{\sf T}$ where $\bf X$ is composed by ${\bf x}_i-{\pmb \mu}$.
$$
{\bf X}=\left[\begin{matrix}
	| &\\
	{\bf A} &\\
	| &
\end{matrix}\right]
\left[\begin{matrix}
	- &\Theta &-\\
	& &
\end{matrix}\right]
$$

### PCA Summary

- Customary to center and scale a data set so that it has zero mean and unit variance along each feature;
- Typically select $k$ such that residual error is small;
- PCA is a good idea when:
  - the data forms a single point cloud in space;
  - the data is approximately Gaussian, or some other elliptical distribution;
  - low-rank subspaces capture the majority of the variation.
- **Big picture**: PCA generates a low-dimensional embedding of the data
  - Euclidean structure of data is approximately preserved;
  - Distances between points are roughly the same before and after PCA;

## Multidimensional Scaling

There are situation for which Euclidean distance is not appropriate.

Suppose we have access to a **dissimilarity matrix** ${\bf D} = [d_{ij}]\in \mathbb{R}^{n\times n}$ and some distance function $\rho$, then $\bf D$ satisfies:
$$
\forall i,j\quad d_{ij}\geq 0 \quad d_{ij}=d_{ji}\quad d_{ii}=0
$$
In particular, triangle inequality is not required.

### Definition

> Multidimensional Scaling (MDS) aims to find $\rho\in \mathbb{R}^k$,  $k \leq d$ and $\{{\bf x}_i\}_{i=1}^n \in \mathbb{R}^k$ such that $\rho ({\bf x}_i,{\bf x}_j)\approx d_{ij}$;

- In general, perfect embedding into the desired dimension does not exist;
- Many variants of MDS based on choice of $\rho$, whether $\bf D$ is completely known or not.

Note that we are looking for a **new representation** of the original data, which means that the original data are only known through $\bf D​$.

There are two types of MDS:

1. **Metric MDS**: try to ensure that $\rho ({\bf x}_i,{\bf x}_j) \approx d_{ij}​$;
2. **Non Metric MDS**: try to ensure that $d_{ij}\leq d_{\ell m} \Rightarrow \rho ({\bf x}_i,{\bf x}_j) \leq \rho ({\bf x}_{\ell},{\bf x}_m)$. In another word, try to preserve the relative distances).

### Euclidean Embedding

Assume $\bf D​$ is completely know (no missing entry) and $\rho ({\bf x}, {\bf y})\triangleq \|{\bf x}-{\bf y}\|_2​$.

- Form ${\bf B} = -\frac{1}{2}{\bf H}{\bf D}^2{\bf H}$ where ${\bf H}\triangleq {\bf I}-\frac{1}{n}{\bf 1}{\bf 1}^{\sf T}$ where ${\bf 1}^{\sf T}=[1,1,\dots,1]$;
- Compute eigen decomposition ${\bf B}={\bf V}{\pmb \Lambda}{\bf V}^{\sf T}$;
- Return ${\pmb \Theta} = ( {\bf V}_k {\pmb \Lambda}_k^{\frac{1}{2}} )^{\sf T}​$,where ${\bf V}_k​$ consists first $k^{\rm th}​$ columns of ${\bf V}​$, ${\pmb \Lambda}_k​$ upper left $k\times k​$ submatrix of $\pmb \Lambda​$.

**Proof**

Assume we have exact embedding ${\bf X}=[{\bf x}_1,\dots,{\bf x}_n]\in \mathbb{R}^{k\times n}​$.

Consider ${\bf D}^2 = [d_{ij}^2]\in \mathbb{R}^{n\times n}$ where $d_{ij}^2 = \|{\bf x}_i - {\bf x}_j\|_2^2 = \|{\bf x}_i\|^2 + \|{\bf x}_j\|^2-2{\bf x}_i^{\sf T}{\bf x}_j​$.

We can write it in the matrix form:
$$
{\bf D}^2 = {\bf b}{\bf 1}^{\sf T} + {\bf 1}{\bf b}^{\sf T} - 2{\bf X}^{\sf T}{\bf X}
\quad \text{ where }\quad
{\bf b}=\left[ \begin{matrix} \|{\bf x}_1\|_2^2 & \cdots & \|{\bf x}_n\|_2^2 \end{matrix}\right]^{\sf T}\in \mathbb{R}^n
$$

${\bf b}{\bf 1}^{\sf T}$ replicates $\bf b$ over the columns and ${\bf 1}{\bf b}^{\sf T}$ does the similar thing. Change the positions of elements in (20) we have:
$$
{\bf X}^{\sf T}{\bf X} = \frac{1}{2} \left({\bf b}{\bf 1}^{\sf T} + {\bf 1}{\bf b}^{\sf T} - {\bf D}^2\right)
$$
Assume that data is centered around zero i.e. we work with $\tilde{\bf X}\triangleq {\bf X}{\bf H}$:
$$
\begin{aligned}
\tilde{\bf X}^{\sf T}\tilde{\bf X} &= \frac{1}{2} \left({\bf H}^{\sf T}{\bf b}{\bf 1}^{\sf T}{\bf H} + {\bf H}^{\sf T}{\bf 1}{\bf b}^{\sf T}{\bf H} - {\bf H}^{\sf T}{\bf D}^2 {\bf H}\right) \\
&= -\frac{1}{2} {\bf H}^{\sf T}{\bf D}^2 {\bf H}\\
&= -\frac{1}{2} {\bf H} {\bf D}^2 {\bf H} \in \mathbb{R}^{n\times n}
\end{aligned}
$$
**Eckart-Young Theorem**

> The above algorithm returns the best rank $k$ approximation in the sense that it minimizes $\|{\pmb \Theta}^{\sf T}{\pmb \Theta}-{\bf B}\|_2$ and $\|{\pmb \Theta}^{\sf T}{\pmb \Theta}-{\bf B}\|_F$.

Details could be found [here](<https://en.wikipedia.org/wiki/Low-rank_approximation>).

### Relationship to PCA

Suppose we have ${\bf X}\in \mathbb{R}^{d\times n}$ and ${\bf D}\in \mathbb{R}^{n\times n}$ with $d_{ij}\triangleq\| {\bf x}_i - {\bf x}_j \|_2$:

- PCA computes an eigen decomposition of ${\bf S}=\sum_{i=1}^N{\bf x}_i{\bf x}_i^{\sf T}={\bf X}{\bf X}^{\sf T}\in \mathbb{R}^{d\times d}​$.
  - Equivalent to computing the SVD of ${\bf X} = {\bf U}{\Sigma}{\bf V}^{\sf T}​$;
  - New representation computed as ${\pmb \theta}_i = {\bf U}_k^{\sf T}{\bf x}_i$, where ${\bf U}_k \in \mathbb{R}^{d\times k}$.
- MDS computes an eigen decomposition of ${\bf X}^{\sf T}{\bf X}\in \mathbb{R}^{n\times n}​$.
  - ${\bf X}^{\sf T}{\bf X} = {\bf V}{\Sigma}^2{\bf V}^{\sf T}$;
  - New representation computed as ${\pmb \Theta} =\left( {\bf V}_k \Sigma_k \right)^{\sf T}$.

The results from PCA and MDS are the same if all data are observed.

**Subtle differences:**

- PCA gives us access to $\bf A$ and $\pmb \mu$: we can extract features and reconstruct approximations;
- Need $\bf X$ to recover $\bf A$;
- In MDS we cannot directly extract features and compute ${\bf A}({\bf x}-{\pmb \mu})$.

### Add points to MDS

Assume we have access to $\bf D$ and want to add a new point $\bf x$ to our embedding, define
$$
\begin{equation*}

d_{\pmb \mu} \triangleq \frac{1}{n} {\bf D}^2{\bf 1}
\qquad
d_{\bf x} \triangleq \left[
\begin{matrix}
	\|{\bf x} - {\bf x}_1 \|_2^2 \\
	\vdots \\
	\|{\bf x} - {\bf x}_n \|_2^2
\end{matrix} \right]

\end{equation*}
$$
Then ${\bf A}^{\sf T}({\bf x}-{\pmb \mu}) = ({\pmb \Theta}^{\dagger})^{\sf T} \left( \frac{1}{2}(d_{\bf x}-d_{\pmb \mu}) \right)$ where ${\pmb \Theta}^{\dagger}$ consists of first $k$ columns of ${\bf V}{\Sigma}^{-1}$ from the SVD $\tilde{\bf X}={\bf U}{\Sigma}{\bf V}^{\sf T}$.

### Extensions

Classical MDS minimizes the loss function $\|{\bf X}^{\sf T}{\bf X}-{\bf B}\|_F$, but many other choices for loss functions exists.

A common other choice is **stress function** $\sum_{i,j}w_{i,j}(d_{ij}-\|{\bf x}_i - {\bf x}_j\|_2)^2$ where $w_{i,j}$is fixed $w_{i,j}\in \{ 0,1 \}$ handles missing data and $w_{i,j}=\frac{1}{d_{ij}^2}$ penalizes error on nearby points.

**Nonlinear embeddings**:

- High-dimensional data sets can have nonlinear structure that not captured via linear methods;
- Kernelize PCA and MDS with non-linear ${\pmb \Phi}:\mathbb{R}^d \to \mathbb{R}^k​$;
- Use PCA on ${\pmb \Phi}({\bf X}){\pmb \Phi}({\bf X})^{\sf T}$ or MDS on ${\pmb \Phi}({\bf X})^{\sf T}{\pmb \Phi}({\bf X})​$.

## Reference

[[1]](<https://bloch.ece.gatech.edu/ece6254sp19/slides/lecture22-slides.html#/title-N>) M. Bloch, (2019, March 28). Dimensionality reduction.

[[2]](https://bloch.ece.gatech.edu/ece6254sp19/slides/lecture23-slides.html#/title-N) M. Bloch, (2019, April 2). PCA and MDS.

[[3]](http://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm) MIT. BE. 400 / 7.548. Singular Value Decomposition (SVD) tutorial.

