---
title: On the Numerical Instablity of Conjugate Gradient
date: 2019-02-15 23:34:26
tags: Machine learning
mathjax: true
---

## 问题描述
<!-- more -->
在tensorflow计算图中实现的Conjugate gradient算法，一开始迭代数值就变成nan。

这个问题是在实现natural gradient optimization的时候发现的，代码实现的思路严格参照了[Wikipedia上有关conjugate gradient的讲解](https://en.wikipedia.org/wiki/Conjugate_gradient_method)，由于代码真正运行是在静态的计算图中，运行时也很难debug出到底出问题的是哪一步。

最后问题解决的办法也非常暴力——在屏幕上打印出所有的变量一一检查，逐层定位问题。经检查，最有可能的两个问题是：
- Conjugate gradient算法实现中可能会出现分母为零的情况
- 计算KL divergence时出现浮点数下溢，导致log输出NaN

理论上来讲这种情况是不会发生的，因为information geometry optimization的核心度量Fisher information matrix一般来讲都是非奇异的。给定$N$个不同的样本，FIM的rank不小于$N\times{\dim(\mathcal{Z})}$，其中$\dim(\mathcal{Z})$为模型输出空间概率分布本身的维度。这也就是说我们只要把batch size给到足够大，FIM不会出现奇异的情况

但实际代码运行的时候，由于一些数值上的不稳定性，FIM奇异还是有可能会出现

## Code implementation

### Conjugate gradient in TensorFlow static computational graph

Conjugate gradient的实现确实是比较复杂，想要将其写在TensorFlow的静态计算图内部还是比较麻烦，遇到的坑会比用numpy实现多不少，以下先放代码

```python
def hessian_vector_product(x, grad, variable):
    kl_grad_prod = tf.reduce_sum(grad * x)
    return tf.stop_gradient(tf.gradients(kl_grad_prod, variable)[0])


def build_conjugate_gradient(x, kl_grad, variable, n_iter=10, func_Ax=hessian_vector_product):
    """build_conjugate_gradient
    :param x: type tf.Tensor, the initial value of x
    :param kl_grad: type tf.Tensor, the gradient of the objective
    :param variable: type tf.Variable
    :return: the converged conjugate gradient vector \tilde{x} = H^{-1}x

    Fixed number of iterations in the inner loop
    """
    x = tf.stop_gradient(x)
    r = x - func_Ax(x, kl_grad, variable)
    p = tf.stop_gradient(r)
    r_dot_r = tf.reduce_sum(tf.square(r))
    for k in range(n_iter):
        Ap = func_Ax(p, kl_grad, variable)
        p_dot_Ap = tf.reduce_sum(p * Ap)
        alpha = r_dot_r / (p_dot_Ap + EPSILON)
        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_r_new = tf.reduce_sum(tf.square(r))
        beta = r_dot_r_new / (r_dot_r + EPSILON)
        r_dot_r = r_dot_r_new
        p = r + beta * p
    return x
```

Ideas worth noting:
- The conjugate gradient does not require direct access to the explicit form of the matrix, only matrix-vector-product is needed. Let $\hat{F}\in{\mathbb{R}^{N\times{N}}}$ be the FIM estimated with a batch of data, and let $v\in{\mathbb{R}^{N}}$ be an arbitrary vector, then $\hat{F}^{-1}v=\nabla_{\theta}(v^{T}\nabla_{\theta}D_{KL}(\pi_{\text{old}}||\pi))$.
- When calculating the matrix-vector-product $\hat{F}^{-1}v$, remember to block the gradient from vector $v$ using `tf.stop_gradient` in every iteration. Your gradient should **only come from the KL divergence term**.
- There are devision operations for variables `alpha` and `beta`. Remember to add a small number (`EPSILON`) for the denominator in case of NaN output.

github上也可以找到其他的基于TensorFlow静态计算图的conjugate gradient实现，[其中一个印象比较深刻的实现方法](https://github.com/steveKapturowski/tensorflow-rl/blob/master/algorithms/trpo_actor_learner.py)是将while循环判断也写在了计算图的内部，用`tf.while_loop`实现，代码如下

```python
def _conjugate_gradient_ops(self, pg_grads, kl_grads, max_iterations=20, residual_tol=1e-10):
    i0 = tf.constant(0, dtype=tf.int32)
    loop_condition = lambda i, r, p, x, rdotr: tf.logical_and(
        tf.greater(rdotr, residual_tol), tf.less(i, max_iterations))

    def body(i, r, p, x, rdotr):
        fvp = utils.ops.flatten_vars(tf.gradients(
            tf.reduce_sum(tf.stop_gradient(p)*kl_grads),
            self.policy_network.params))
        z = fvp + self.cg_damping * p
        alpha = rdotr / (tf.reduce_sum(p*z) + 1e-8)
        x += alpha * p
        r -= alpha * z
        new_rdotr = tf.reduce_sum(r*r)
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        new_rdotr = tf.Print(new_rdotr, [i, new_rdotr], 'Iteration / Residual: ')
        return i+1, r, p, x, new_rdotr

    _, r, p, stepdir, rdotr = tf.while_loop(
        loop_condition,
        body,
        loop_vars=[i0,
                    pg_grads,
                    pg_grads,
                    tf.zeros_like(pg_grads),
                    tf.reduce_sum(pg_grads*pg_grads)])
    fvp = utils.ops.flatten_vars(tf.gradients(
        tf.reduce_sum(tf.stop_gradient(stepdir)*kl_grads),
        self.policy_network.params))
    shs = 0.5 * tf.reduce_sum(stepdir*fvp)
    lm = tf.sqrt((shs + 1e-8) / self.max_kl)
    fullstep = stepdir / lm
    neggdotstepdir = tf.reduce_sum(pg_grads*stepdir) / lm
    return fullstep, neggdotstepdir
```

### Model definition and environment interaction

调了一晚上的bug，根源就在这部分代码中。在[jupyter notebook](https://github.com/ChenShawn/advanced_policy_gradient_methods/blob/master/debug.ipynb)中打印出了所有变量的实际值后，发现一些奇怪的现象：

- 问题出在从普通的gradient转成natural gradient的过程中，普通gradient数值没有问题，natural gradient时常数值爆炸，或者直接出现NaN
- 搭了一个小网络来测试conjugate gradient实现的正确与否，发现偶尔会出现FIM奇异的情况，推测是numerical instability所致
- 进一步的测试中发现，每一步的迭代中并不总是会出现FIM奇异。相对而言较大、且靠近底层的参数不容易出现FIM奇异，参数量较小且靠近输出层的参数容易出现NaN，当时最频繁出现NaN的参数，就是网络最后一层的bias

猜测这种numerical instability可能是由于KL divergence不稳定导致的，所以干脆把旧的policy网络和新的policy网络写成同一个，设置`reuse=True`的来共享网络参数，这样KL divergence就会始终为0，毕竟我们需要的只是KL divergence的gradient和Hessian而已。

后续实验中发现这样写会使得网络迭代更新速度缓慢，且容易收敛到sub-optimal点，有关这个问题之后如有新的发现再来更新这里的内容吧

```python
def collect_multi_batch(env, agent, maxlen, batch_size=64, qsize=5):
    """collect_multi_batch
    See collect_one_trajectory docstring
    :return: three lists of batch data (s, a, r)
    """
    que = []
    s_init = env.reset()
    que.append(s_init[None, :])
    for it in range(qsize - 1):
        st, r, done, _ = env.step([-0.99])
        que.append(st[None, :])
    # Interact with environment
    buffer_s, buffer_a, buffer_r = [], [], []
    for it in range(maxlen):
        s = np.concatenate(que, axis=-1)
        a = agent.choose_action(s)
        buffer_s.append(s)
        s, r, done, _ = env.step(a)
        que.pop(0)
        que.append(s[None, :])
        buffer_a.append(a[None, :])
        r = (r + 0.3) * 2.0
        buffer_r.append(r)
        if done:
            break
    # Accumulate rewards
    discounted = 1.0
    last_value = model.value_estimate(s)
    discounted_r = []
    for it in range(len(buffer_r) - 2, -1, -1):
        last_value = r + args.gamma * last_value
        discounted_r.append(last_value)
    state_data, action_data, reward_data = [], [], []
    for it in range(0, maxlen, batch_size):
        if it >= len(buffer_s):
            break
        states_array = np.concatenate(buffer_s[it: it + batch_size], axis=0)
        actions_array = np.concatenate(buffer_a[it: it + batch_size], axis=0)
        rewards_array = np.array(discounted_r[it: it + batch_size], dtype=np.float32)[:, None]
        # rewards_array = np.clip(rewards_array, -1.0, 5.0)
        state_data.append(states_array)
        action_data.append(actions_array)
        reward_data.append(rewards_array)
    return state_data, action_data, reward_data


class TNPGModel(object):
    def __init__(self, v_lr, pi_lr, model_dir, delta=1e-3):
        self.state = tf.placeholder(tf.float32, [None, 10], name='state')
        self.action = tf.placeholder(tf.float32, [None, 1], name='action')
        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')

        # Advantage function definition
        print(' [*] Building advantage function...')
        kwargs = {'kernel_initializer': tf.orthogonal_initializer()}
        with tf.variable_scope('value'):
            h1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu, name='h1', **kwargs)
            self.value = tf.layers.dense(h1, 1, activation=None, name='value', **kwargs)
            self.advantage = self.reward - self.value

            self.v_loss = tf.reduce_mean(tf.square(self.advantage))
        v_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')
        self.v_train = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss, var_list=v_vars)

        # Policy function definition
        print(' [*] Building policy function...')
        self.policy, pi_vars = build_gaussian_network(self.state, 1, scope='policy')
        old_policy, old_vars = build_gaussian_network(self.state, 1, scope='policy', trainable=False, reuse=True)
        with tf.name_scope('policy_ops'):
            # self.assign_op = [old.assign(new) for old, new in zip(old_vars, pi_vars)]
            self.sample_op = self.policy.sample(1)
        with tf.name_scope('surrogate_loss'):
            ratio = self.policy.prob(self.action) / old_policy.prob(self.action)
            surrogate = ratio * self.advantage
            self.pi_loss = -tf.reduce_mean(surrogate)

        # Convert Adam gradient to natural gradient
        print(' [*] Building natural gradient...')
        with tf.variable_scope('policy_optim'):
            kl = tf.distributions.kl_divergence(old_policy, self.policy)
            optim = tf.train.AdamOptimizer(pi_lr)
            pi_grads_and_vars = optim.compute_gradients(surrogate, var_list=pi_vars)
            pi_grads = [pair[0] for pair in pi_grads_and_vars]
            kl_grads = tf.gradients(kl, pi_vars)

            conj_grads = []
            for grad, kl_grad, var in zip(pi_grads, kl_grads, pi_vars):
                conj = build_conjugate_gradient(grad, kl_grad, var)
                nat_grad = tf.sqrt((2.0 * delta) / (tf.reduce_sum(grad * conj) + EPSILON)) * conj
                conj_grads.append((nat_grad, var))
            self.pi_train = optim.apply_gradients(conj_grads)

        # Summaries definition
        print(' [*] Building summaries...')
        model_variance = tf.reduce_mean(self.policy._scale)
        self.sums = tf.summary.merge([
            tf.summary.scalar('max_rewards', tf.reduce_max(self.reward)),
            tf.summary.scalar('mean_advantage', tf.reduce_mean(self.advantage)),
            tf.summary.scalar('pi_loss', self.pi_loss),
            tf.summary.scalar('v_loss', self.v_loss),
            tf.summary.scalar('model_variance', model_variance)
        ], name='summaries')

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        print(' [*] Model built finished')
        _, self.counter = load(self.sess, model_dir)


    def choose_action(self, s):
        a = self.sess.run(self.sample_op, feed_dict={self.state: s})
        return a[0, :, 0]


    def update(self, s, a, r, v_iter, pi_iter, writer=None, counter=0):
        feed_dict = {self.state: s, self.action: a, self.reward: r}
        # self.sess.run(self.assign_op)
        # update policy
        for _ in range(pi_iter):
            self.sess.run(self.pi_train, feed_dict=feed_dict)
        # update value function
        for _ in range(v_iter):
            self.sess.run(self.v_train, feed_dict=feed_dict)
        if writer is not None:
            sumstr = self.sess.run(self.sums, feed_dict=feed_dict)
            writer.add_summary(sumstr, counter)
```