---
title: Oral Presentation Notes
date: 2019-01-18 08:29:45
tags: Personal
mathjax: true
---

## The presentation notes used in AAAI-2019 oral presentation

<!--more-->
Good morning/afternoon, everyone. It’s honored to be here to present our work “the adversarial attack and detection under the Fisher information metric”. My name is Zhao Chenxiao. It’s joint work with Prof. Fletcher, Yu Mixue, Prof Peng, Prof Zhang, and Prof Shen. 

So let’s start our topic with the notion of adversarial examples. A wide range of researches point out that deep learning models are vulnerable to the adversarial attack. Some specifically designed noise added on the input can be imperceptible for human, but make the models completely failed, and the phenomenon is quite general for various tasks.

To better understand the adversarial examples, a fundamental problem is how to measure the vulnerability of the deep learning models. There are many ways to generate adversarial examples, but is there any way to describe the models’ robustness to these attacks under the same framework? One of the ideas is to use the worst case perturbations. If we can quantify it or optimize a surrogate of it, the robustness can be progressively improved by adversarial training. Another stream of study assumes that the distributions of the adversarial examples and normal samples are different. Using some characteristics that can describe the difference between them, this leads to the approach of adversarial detection.

**Our approach combines both the adversarial attack and detection under the same framework.** We assume that there exists a metric tensor in the output space, and the mapping of the neural network from the input space to the output space makes a pullback. That means we have a metric that can be used to characterize how the perturbation on the input will influence the output. Given an input sample, if we use \eta to denote the perturbation vector, the variation of the output in the tangent space will be given by this equation. As can be seen, they are related by both the Jacobian matrix and the metric tensor.

So let’s skip the mathematical definitions first, say, if we already have a well-defined metric tensor $G^{x}$ in the data space, how do we formulate the objective function for adversarial perturbations? It is easy to figure out that the vulnerability can be defined by the quadratic form, where we constrain the norm of the perturbation on an epsilon sphere. The solution is also very simple: the optimal adversarial perturbation $\eta$ is the greatest eigenvector. So let’s get back to the question: **how to define the metric tensor G?**

**Intrinsically, why should the metric for the output predictions be non-linear?** The basic idea is that, the model should be regarded as not only a function mapping, but also a probabilistic model. The set of all probability distributions conditioned on the model parameter is a manifold. So we don’t use Euclidean distance to measure to distance between distributions. 

Among all the distance measures, the metric defined by Fisher information has been proved to be the unique measure that is invariant to reparameterization. It is obviously positive semi-definite by definition. There are many other theoretical benefits. Please refer to this reference.

For adversarial attacks, the input $x$ is the only changeable variable. We can obtain the following form of FIM using some exchange of variables. But more specifically, when compute the matrix numerically, what is $p(y|x)$ here? Basically we have two choices, the model distribution or empirical distribution. The model distribution can not be directly observed but really reflects some intrinsic features of the model, like uncertainty. The empirical distribution is just a multi-dimensional discrete distribution given by the last softmax layer. 

Using different definitions, here are some ways to compute the matrix. The vanilla approach is to use the explicit form of the Jacobian, and this would be computationally expensive. Another way is to compute the second derivative of the KL divergence. The adversarial training based on this theory is known as the virtual adversarial training.

So our solution is to use the empirical distribution to compute the Fisher, which is the expectation over the outer product. This can give us many engineering benefits. First of all, it is easy to compute, providing one is already calculating the gradient. Another advantage is that this form make it a lot easier to calculate the matrix-vector-product without access to the explicit form of the matrix, and some eigen-decomposition methods, incuding power iteration and Lanczos method, only require matrix-vector-product to compute the eigenvalues and eigenvectors.

**When the algorithm is applied on large datasets. We still have two problems.** The first one is that if input sample X is high dimensional, the matrix G will be too big that one can not even load it into memory. As mentioned before, this can be solved using power iteration or Lanczos method. Both of them only need to matrix-vector-product instead of the explicit matrices. One thing to note is that the Lanczos algorithm can calculate a group of eigenvalues and the eigenvectors instead of just the maximum one, and it's particularly fast for sparse matrices.

Another problem is that for datasets like ImageNet, there are 1000 classes. In this case it will be hard to implement the expectation in parallel. Our solution is to use Monte-Carlo sampling from the empirical distribution $r$. 

We empirically find the output probabilities given by ImageNet models is long tail, so using only about 1/5 times of sampling is sufficient to guarantee the performance.

Here are some comparisons of the attack using our algorithm, where we have compared our results with two gradient based methods. The horizontal axis is the fooling rate. The interesting part is that our method often gets higher fooling rate in one-step attack case. But if you use the algorithm iteratively the results get very similar to the results of the iterative variant of the fast gradient method.

This figure shows the relationship between the vulnerability of the models and the eigenvalues. The vertical axis is the $\ell_{2}$ norm of the least perturbations, which is obtained via binary search along the direction of our attack. The horizontal axis is the logarithm of eigenvalues. As we can see the empirical evidence shows a linear relationship between them: The larger the eigenvalues are, the more vulnerable the model is in the corresponding subspaces.

This suggests that we can use the eigenvalues as the vulnerability characteristic. This figure further shows the eigenvalues of adversarial examples and normal samples are distributed differently. In the left figure, we can see adding Gaussian noise on samples does not really change the eigenvalues of them, but the adversarial examples are more likely to have larger eigenvalues. In the right figure, we find that if we modify the samples along the direction of the perturbation, most of the eigenvalues of the samples are rapidly increasing with the increasing of the perturbation size.

Based on the observation, we use an auxiliary classifier to distinguish the adversarial examples. For practical reasons we use the logarithm of eigenvalues as features. We use the aforementioned Lanczos method to get a group of eigenvalues instead of just the maximum one. Motivated by the previous literature, we also add noisy samples into the positive training set. This works very well on enhancing the generalization ability of the classifier.

Some of our evaluations are shown here. We compare our method with the kernel density estimation and Bayesian uncertainty. We observe that our method achieves good performance on MNIST. 

The performance drops a little bit on CIFAR-10, but we can see the method still works nice when recognizing the three attacks on the right side.

This form shows the generalization ability of our classifier trained using only one kind of attacks. As we can see here, our proposed method generalizes well on $\ell_{2}$ and $\ell_{\infty}$ norm attacks, but it fails to generalize to JSMA, which is an L0 norm case. As we know the $\ell_{0}$ norm space is discrete, making its topological structure a lot different with the other norms. From my personal point of view, it will be an interesting future work to extend the idea to a more general framework.

That’s all. Thank you.