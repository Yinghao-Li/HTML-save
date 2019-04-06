---
layout:     post
title:      Dimensionality Reduction
subtitle:   Class Notes of Gatech ECE 6254
date:       2019-04-05
author:     Yinghao Li
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Machine Learning
    - Dimension
    - Gatech
---



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
\tag{1}
r(j) \triangleq \frac{1}{N}\sum_{i=1}^N\mathbb{i}\left\{y_i\neq h(x_{i,j})\right\}
$$

​	for some classifier $h$;

- Two-sample t-test statistics:

$$
\tag{2}
r(j)\triangleq \frac{\bar{x}_j^{(+)} - \bar{x}_j^{(-)}}{s/\sqrt{n}}
$$

​	where $\bar{x}_j^{(\pm)}​$ are class means for feature $j​$; $s​$ is the polled sample standard deviation.

- Margin:

$$
\tag{3}
r(j)\triangleq \min_{k,\ell}\left|x_j^{\mathbb{I}(y_k=1)}-x_j^{\mathbb{I}(y_l=-1)}\right|
$$

### Ranking in Regression

- Correlation coefficient:

$$
\tag{4}
r(j)\triangleq |\rho(j)| \text{ with } \rho(j)\triangleq\frac{{\rm Cov}(x_j,y)}{\sqrt{{\rm Var}(x_j){\rm Var}(y)}}
$$

- Mutual information:

$$
\tag{5}
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

## Reference

[[1]](<https://bloch.ece.gatech.edu/ece6254sp19/slides/lecture22-slides.html#/title-N>) M. Bloch, (2019, March 28). Dimensionality reduction.



