---
layout:     post
title:      Regression and Regularization
subtitle:   Class Notes of Gatech ECE 6254
date:       2019-04-01
author:     Yinghao Li
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Machine Learning
    - Regression
    - Regularization
    - Gatech
---

# Regression and Regularization

[TOC]

## Regression

>In classification, $y_i \in \mathcal{Y} \subset \mathbb{R}$ with $|\mathcal{Y}| \triangleq K < \infty$ while in regression $y_i \in \mathcal{Y} \subseteq \mathbb{R}$.

### Least Square Regression

Least square regression corresponds to the situation in which the loss function is sum of square errors.
$$
{\rm SSE}({\pmb \beta}, \beta_0) \triangleq \sum_{i=1}^N(y_i-{\pmb \beta}^{\sf T}-\beta_0)^2
\tag{1}
$$

#### Matrix Representation

If we add bias term into $\bf X$ and define
$$
{\pmb \theta} \triangleq \left[
    \begin{matrix}
    \beta_0 \\ \beta_1 \\ \vdots \\ \beta_d
    \end{matrix}
    \right]
$$

Then equation (1) can be written as
$$
{\rm SSE}({\pmb \beta}, \beta_0) \triangleq \|{\rm y - X{\pmb \theta}}\|_2^2
\tag{2}
$$

#### Analytical Solution

If ${\bf X}^{\sf T}{\bf X}$ is non-singular, the close form analytical solution to equation (2) is
$$
\hat{\pmb \theta} = \left({\bf X}^{\sf T}{\bf X}\right)^{-1}{\bf X}^{\sf T}{\bf y}
\tag{3}
$$

The **proof** is similar to the regularized form below.

Even if ${\bf X}^{\sf T}{\bf X}​$ is singular, we can use SVD to get the approximate solution.

We can also apply a non-linear feature map $\Phi : \mathbb{R}^d \to \mathbb{R}^\ell : {\bf x} \mapsto \Phi ({\bf x})$ to get some additional benefits.

## Regularization

>Overfitting occurs as the number of features $d$ begins to approach the number of observations $N$

An approach to deal with overfitting is **regularization**.

### Thykonov Regularization

The key idea is to introduce a penalty term to limit the value of vector $\pmb \theta$.

$$
\tag{4}
\hat{\pmb \theta} = \underset{\pmb \theta}{\operatorname{argmin}} \|{\bf y-X{\pmb \theta}}\|_2^2+\|{\bf \Gamma} {\pmb\theta}\|_2^2
$$
where ${\bf \Gamma}\in \mathbb{R}^{(d+1)\times(d+1)}$.

#### Analytical Solution

The minimizer of the least-square problem with Thykonov regularization is:
$$
\hat{{\pmb \theta}} = \left({\bf X}^{\sf T}{\bf X} + {\bf \Gamma}^{\sf T}{\bf \Gamma} \right)^{-1}{\bf X}^{\sf T}{\bf y}
\tag{5}
$$

> **Proof**:
> $$
> \begin{aligned}
> & \frac{\partial}{\partial {\pmb \theta}}\left(\|{\bf y-X{\pmb \theta}}\|_2^2+\|{\bf \Gamma} {\pmb\theta}\|_2^2 \right) \\
> =& \frac{\partial}{\partial {\pmb \theta}}\left(({\bf y-X{\pmb \theta}})^{\sf T}({\bf y-X{\pmb \theta}}) + {\pmb\theta}^{\sf T}{\bf \Gamma}{\bf \Gamma}{\pmb\theta} \right) \\
> =& \frac{\partial}{\partial {\pmb \theta}}\left(({\bf y-X{\pmb \theta}})^{\sf T}({\bf y-X{\pmb \theta}}) + {\pmb\theta}^{\sf T}{\bf \Gamma}{\bf \Gamma}{\pmb\theta} \right) \\
> =& \frac{\partial}{\partial {\pmb \theta}} \left( {\bf y}^{\sf T}{\bf y} - 2{\pmb \theta}^{\sf T}{\bf X}{\sf T}{\bf y} + {\pmb\theta}^{\sf T}{\bf X}{\bf X}{\pmb\theta} + {\pmb\theta}^{\sf T}{\bf \Gamma}{\bf \Gamma}{\pmb\theta} \right) \\
> =& 2{\bf X}^{\sf T}{\bf X}{\pmb \theta} + 2{\bf \Gamma}^{\sf T}{\bf \Gamma}{\pmb \theta} - 2{\bf X}^{\sf T}{\bf y} = 0
> \end{aligned}
> $$
> Solving this equation we get
> $$
> \hat{{\pmb \theta}} = \left({\bf X}^{\sf T}{\bf X} + {\bf \Gamma}^{\sf T}{\bf \Gamma} \right)^{-1}{\bf X}^{\sf T}{\bf y}
> $$
> For special case ${\bf \Gamma} = \sqrt{\lambda}{\bf I}$ for $\lambda > 0$, we obtain
> $$
> \tag{6}
> \hat{{\pmb \theta}} = \left({\bf X}^{\sf T}{\bf X} + \lambda{\bf I} \right)^{-1}{\bf X}^{\sf T}{\bf y}
> $$
> 

#### Ridge Regression

Ridge regression is slightly different from above which does not penalize $\beta_0$. In ridge regression,
$$
{\bf \Gamma} \triangleq \left[
    \begin{matrix}
    0 & 0 &\cdots & 0 \\
    0 & \sqrt{\lambda} & \cdots & \lambda\\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \vdots & \sqrt{\lambda}
    \end{matrix}
    \right]
$$

Thykonov regularization can also be regarded as a constrained optimization problem. Solving Thykonov regularization is equivalent to solving

$$\tag{7}
\underset{\pmb \theta}{\operatorname{argmin}} \|{\bf y-X{\pmb \theta}}\|_2^2 \text{ such that } \|{\bf \Gamma} {\pmb\theta}\|_2^2 \leqslant \gamma
$$

The figure below illustrates the effect of Thykonov regularization in $\mathbb{R}^2​$ assuming ${\bf \Gamma} = {\bf I}​$.

![post-RR-v1](https://raw.githubusercontent.com/Yinghao-Li/Yinghao-Li.github.io/master/img/post-RR/post-RR-v1.PNG)

## Reference

[[1]](<https://bloch.ece.gatech.edu/ece6254sp19/lecture15-v1.pdf>) M. Bloch, (2019, March 22). Lecture 15 - Regression and regularization.

[[2]](<https://bloch.ece.gatech.edu/ece6254sp19/slides/lecture15-slides.html#/title-N>) M. Bloch, (2019, Feburary 26). Regression and Regularization.



