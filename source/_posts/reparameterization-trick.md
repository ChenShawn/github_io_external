---
title: On Implementing the Reparameterization Trick
date: 2019-02-06 14:45:36
tags: Machine learning
mathjax: true
---

第一次接触到reparameterization trick是在variational auto-encder的文章中，由于其中损失函数含有hidden layer真实分布与高斯先验之间的KL divergence项，在实现时将hidden layer重新参数化成一个高斯分布。

<!-- more -->

后来发现reparameterization trick的应用场景远不止VAE这么简单，这种实现上的trick在Bayesian deep learning与reinforcement learning中都有非常广泛的应用。本文基于tensorflow对reparameterization trick的实现做简单总结。

## 1. 什么情况下可以用reparameterization trick？

总体来说，有下面几种情况可以用到reparameterization trick
- 神经网络的某个部分需要参数化成一个概率分布的形式，e.g. Gaussian distribution
- 需要将神经网络的输出进行随机化，i.e. 神经网络不再是一个固定的从输入空间到输出空间的映射关系，而是一个包含了随机性的模型

## 2. VAE Implementation

以下为简化版的VAE示意代码

```python
def build_vae(batch_xs, scope='VAE', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        mu, sigma = build_encoder(batch_xs)
        sampled = tf.random_normal(mu.get_shape().as_list(), mu, sigma)
        output = build_decoder(sampled, batch_xs.get_shape().as_list()[-1])
    return output, mu, sigma

def build_encoder(input_op):
    with tf.variable_scope('encoder'):
        h1 = tf.layers.dense(input_op, 256, activation=tf.nn.relu, use_bias=True)
        h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu, use_bias=True)
        mu = tf.layers.dense(h2, 50, activation=None, use_bias=True)
        sigma = tf.layers.dense(h2, 50, activation=tf.softplus, use_bias=True)
    return mu, sigma

def build_decoder(input_op, output_dim):
    with tf.variable_scope('decoder'):
        h1 = tf.layers.dense(input_op, 128, activation=tf.nn.relu, use_bias=True)
        h2 = tf.layers.dense(h1, 256, activation=tf.nn.relu, use_bias=True)
        return tf.layers.dense(h2, output_dim)
```

## 3. tf.distributions

tensorflow中提供了一个tf.distributions类，可以比较方便的对计算图中的随机概率分布进行建模，其详细API可以参考[官方文档](..)。

最核心的成员函数总结如下

- sample(shape): 输入一个采样维度shape，返回一个可以从概率分布中采样的sampler op
- prob(x): 输入一个shape为[batch_size, 1]的tf.Tensor或np.array，返回概率密度函数在$x$点的值
- cdf(x): 输入一个shape为[batcg_size, 1]的tf.Tensor或np.array，返回累计分布函数在$x$点的值，i.e., Let $p(t)$ be the pdf, then the returned op $\int_{-\infty}^{x}p(t)dt$