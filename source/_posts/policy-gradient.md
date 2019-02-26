---
title: Advanced Policy Gradient
date: 2019-02-06 14:17:33
tags: Machine learning
mathjax: true
---

本文的主要内容为CS294-112课程中『Advanced Policy Gradient』一节的总结与代码实现。
<!-- more -->
代码参考了[莫凡大神的github repo](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)

## Policy performance bound

## Truncated natural policy gradient (TNPG)

## Trust region policy gradient (TRPO)

## Proximal policy gradient (PPO)

```python
class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # Critic (Using advantage function to reduce variance)
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, activation=tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # Actor (Function pi(a|s) following behavior policy)
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -tf.reduce_mean(surr - self.tflam * kl)
            elif METHOD['name'] == 'clip':
                # clipping method, find this yields better performance
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.0 - METHOD['epsilon'], 1.0 + METHOD['epsilon']) * self.tfadv)
                )
            else:
                raise NotImplementedError

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter('log/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
```