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
        sampled = mu + sigma * tf.random_normal(mu.get_shape().as_list(), 0.0, 1.0, dtype=tf.float32)
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

batch_xs = tf.placeholder(tf.float32, [None, INPUT_DIM], name='batch_xs')
rec_xs, mu, sigma = build_vae(batch_xs)

# Loss term definition
rec_loss = tf.reduce_mean(tf.square(rec_xs - batch_xs))
kl_loss = tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1.0, axis=-1)
kl_loss = tf.reduce_mean(kl_loss, name='kl_loss')

loss = rec_loss + kl_loss
```

注意方差标准差一定是大于零的，一般常规做法会使用softplus激活函数来确保它们的非负性。简单来说，softplus函数是对ReLU激活函数的光滑近似，其形式为

$$f(x)=\log(1+e^{x})$$

图片来自[wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus)

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Rectifier_and_softplus_functions.svg/1024px-Rectifier_and_softplus_functions.svg.png" width="350px">
</div>

除此以外，也可以用`tf.square`来保证标准差的非负性

## 3. tf.distributions.Distribution

tensorflow中提供了一个概率分布类，可以比较方便的对计算图中的随机概率分布进行建模，`tf.distributions.Distribution`类是抽象基类，其详细API可以参考[官方文档](https://www.tensorflow.org/api_docs/python/tf/distributions/Distribution)。

最核心的几个成员函数总结如下

- sample(shape): 输入一个采样维度shape，返回一个可以从概率分布中采样的sampler op
- prob(x): 输入一个shape为[batch_size, 1]的tf.Tensor或np.array，返回概率密度函数在$x$点的值
- cdf(x): 输入一个shape为[batcg_size, 1]的tf.Tensor或np.array，返回累计分布函数在$x$点的值，i.e., Let $p(t)$ be the pdf, then the returned op $\int_{-\infty}^{x}p(t)dt$
- log_prob(x): The same as prob(x) except that the returned value is, literally, the logarithm of the pdf on a given point.

## 4. Reparameterization trick in Bayesian deep learning

上面的例子中解释了reparameterization trick在VAE中的应用，其中我们为了将hidden layer的概率分布建模成Gaussian distribution，在具体实现时，直接让神经网络输出Gaussian distribution的$\mu$和$\sigma$，再从这两个参数构成的Gaussian distribution中采样得到randomized hidden layer。

更常见的一种情况是，我们希望模型的输出不仅表达了模型对输入的预测，同时也可以建模模型本身的uncertainty（e.g. 用Gaussian distribution的方差来表达）。使用与上面类似的思路，可以让模型直接输出$\mu$和$\sigma$两个vector，真实的预测值来源于$\mathcal{N}(\mu,\sigma^{2})$的采样。

这种思路非常适合应用于Information geometry optimization (IGO)，原因在于，IGO中的一个基本任务是估计Fisher information matrix，一般的计算方式是这样的

$$G^{\theta}=\nabla_{\theta_{k}}\nabla_{\theta_{k}}D_{KL}(p(x,y|\theta)||p(x,y|\theta_{k}))$$

- 如果不使用这种特殊的参数化方式，那么上式中的KL divergence就只能通过empirical distribution来进行估计，这样做是没有办法对数据本身的uncertainty进行建模的。
- 在神经网络中计算Hessian的复杂度太高了，为什么还是执意使用Hessian of KL的形式？主要的目的是引入conjugate gradient来计算${G^{\theta}}^{-1}g$，where $g$ is the gradient under the Euclidean metric。

接下来放示意代码

```python
def build_model(input_op, pred_dim, reuse=False, scope='model'):
    with tf.variable_scope(scope, reuse=reuse):
        """
        Implement the model definition here
        """
        pass

mu, sigma = build_model(batch_xs, PRED_DIM, reuse=False)
mu_old, sigma_old = build_model(batch_xs, PRED_DIM, reuse=True)
mu_old, sigma_old = tf.stop_gradient(mu_old), tf.stop_gradient(sigma_old)
gaussian = tf.distributions.Normal(mu, sigma)
gaussian_old = tf.distribution.Normal(mu_old, sigma_old)

kl_divergence = tf.distributions.kl_divergence(gaussian_old, gaussian)
```

## 5. Can randomized model learn the uncertainty in reinforcement learning?

建模一个概率分布而非确定的函数映射，这种思路在supervised learning框架下是很直观的，然而如果是在强化学习任务上，随机模型是否真的可以学习到环境中所包含的不确定性呢？

