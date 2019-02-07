---
title: Applications of Fisher Information Matrix
date: 2019-01-19 02:15:13
tags: Machine learning
mathjax: true
---

*未完成，待更新。。。*

*Unfinished, to be updated...*

## 1. Statistical definition
<!--more-->
### 1.1. Informal definition

The following definition is from [the wikipedia of Fisher information](https://en.wikipedia.org/wiki/Fisher_information).

> In mathematical statistics, the **Fisher information** (sometimes simply called **information**) is a way of measuring the amount of information that an observable random variable $X$ carries about an unknown parameter $\theta$ of a distribution that models $X$.

The following is from the "related works" section in [my paper](https://arxiv.org/abs/1810.03806).

> The Fisher information is initially proposed to measure the variance of the likelihood estimation given by a statistical model.
Then the idea was extended by introducing differential geometry to statistics.
By considering the FIM of the exponential family distributions as the Riemannian metric tensor,
Chenstov further proves that the FIM as a Riemannian measure is the only invariant measure for distributions.

### 1.2. Formal definition

Formally, let $p(x|\theta)$ be a probability density function of random variable $X$ conditioned on parameter $\theta$.
The Fisher information matrix of $\theta$, denoted as $G^{\theta}$,
is defined as the variance of the expectation over the derivative of log-likelihood with respect to $\theta$:
$$
G^{\theta}_{ij}=\mathbb{E}_{x|\theta}[(\frac{\partial}{\partial{\theta_{i}}}\log{p(x|\theta)})
                                        (\frac{\partial}{\partial{\theta_{j}}}\log{p(x|\theta)})^{T}] 
$$

Some other forms are

$$
G^{\theta}_{ij}=\frac{\partial^{2}}{\partial{\eta_{i}}\partial{\eta_{j}}}\mathbb{E}_{x|\theta}[\log{p(x|\theta)||p(x|\theta+\eta)}]\\
=\mathbb{D}_{x|\theta}[\frac{\partial}{\partial{\theta}}\log{p(x|\theta)}]\\
=-\mathbb{E}_{x|\theta}[\frac{\partial^{2}}{\partial{\theta_{i}}\partial{\theta_{j}}}\log{p(x|\theta)}]
$$

## 2. Fisher information in practice

TODO: 
- vanilla natural gradient descent
- Natural CMA-ES
- TRPO / PPO / ACKTR / etc.
- Virtual adversarial training

## 3. Are all of the definitions numerically equivalent?

In this section, our goal is to verify that all of the aforementioned forms of Fisher information matrix are not only equivalent to the other in theory, but also match in programming computation.

This might not be a problem at first glance, but it has confused me for a long time. The major question including

- If calculating the expectation over the outer product of gradient is sufficient to obtain the matrix, why do people bother calculating second order derivatives when performing natural gradient descent?
- What the hell is the conditional distribution $p(y,x|\theta)$ in practice? What the hell is the conditional distribution $p(y|x,\theta)$ in practice?
- Are all of the forms numerically stable such that all of them result in the exactly the same result when programming?

Let us first verify that the Hessian of KL divergence is indeed equivalent to the expectation over outer product.

```python
import tensorflow as tf
import numpy as np

def build_net(xs):
    with tf.variable_scope('test', reuse=tf.AUTO_REUSE):
        batch_xs = tf.expand_dims(xs, axis=0)
        hs = tf.layers.dense(batch_xs, 30, activation=tf.nn.relu)
        zs = tf.layers.dense(hs, 3, activation=None)
        probs = tf.nn.softmax(zs, axis=-1)
        return probs, zs

xs = tf.ones((3))
perturb = 0.5 * tf.ones_like(xs)
aprob, alogits = build_net(xs)
bprob, blogits = build_net(xs + perturb)
ys = tf.placeholder(tf.int32, [1, 3])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=alogits, labels=ys)
aloss = tf.log(aprob + 1e-25)
bloss = tf.log(bprob + 1e-25)
kl = tf.reduce_sum(aprob * aloss) - tf.reduce_sum(bprob * bloss)
grad = tf.gradients(tf.reduce_mean(loss), xs)[0]
hessian = tf.hessians(kl, perturb)[0]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    fisher = np.zeros((3, 3), dtype=np.float32)
    p = sess.run(aprob)
    for it in range(3):
        y = np.zeros((1, 3), dtype=np.int32)
        y[0, it] = 1
        g = sess.run(grad, feed_dict={ys: y})
        fisher += p[0, it] * np.matmul(g[:, None], g[None, :])

    fisher2 = -sess.run(hessian)

print(fisher)
print(fisher2)

print(np.linalg.eig(fisher)[0].max())
print(np.linalg.eig(fisher2)[0].max())
```

TODO: add results below

