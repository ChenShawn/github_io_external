---
mathjax: true
title: MCMC基础
tags: Machine learning
date: 2018-05-09 18:08:15
---

*This post is mainly for the testing of MathJax in hexo*

## 1. Random walk
<!--more-->
对于一个无向图 $G(V,E)$ ，给定初始位置结点$ v_{0} $，以均匀概率随机挑选与其相连的一条边，然后移动到该边上的相邻结点，不断重复这一过程，这个过程称为随机游走。

## 2. Markov chain

给定一个离散时间序列随机过程 $\{X_{0},X_{1},X_{2},...,X_{T}\}$，当其满足

$${P}(X_{t+1} = j | X_{0} = i_{0} , X_{1} = i_{1} ,..., X_{t} = i_{t} ) = P( X_{t+1} =j | X_{t} = i_{t})$$

该随机过程为齐次马尔可夫链

<b>总结一句就是：Markov chain具有无记忆性，T时刻的状态包含了T时刻之前所有的信息</b>

## 3. Stationary distribution: random walk on a Markov chain
<b>定义:这里将Markov chain上的概率转移矩阵 ${P}\in{[0,1]^{n\times{n}}}$ 定义为</b>

$$
P=\left(\begin{matrix}
    p_{1,1} &    p_{1,2}  & ...  & p_{1,n} \\
    p_{2,1} &    p_{2,2}  & ...  & p_{2.n} \\ 
    ... &    ...  & ...  & ... \\
    p_{n,1} & p_{n,2}     & ...  & p_{n,n}
\end{matrix}\right)
$$

其中$ p_{i,j} $为从Markov chain的第i个节点转换到第j个节点的概率

将state probability vector定义为 $\pi^{t}=(\pi_{1}^{t},\pi_{2}^{t},...,\pi_{n}^{t})$ ，其中 $\sum_{i=0}^{n}\pi_{i}^{t}=1$ 。则该Markov chain上每一步的随机游走可以写成：

$$ {\pi}^{t+1}={P} \pi^{t} $$

很明显可以看出对于 $i\in{1,2,...,n}$，有 $\sum_{k=1}^{n}{P}_{i,k}=1$，所以每一步转移结束后向量${\pi}$仍然会是概率分布的形式

<b>Stationary distribution 平稳分布</b>

- (Informal) 简单来说，进行完一步随机游走概率转换后，若 ${\pi}^{t-1}={\pi}^{t}$，称${\pi}^{t}$为该随机过程的**平稳分布** (stationary distribution)
- (Strict definition) 懒得打字，链接到[wiki: stationary distribution](https://en.wikipedia.org/wiki/Stationary_distribution)还有[wiki: stationary process](https://en.wikipedia.org/wiki/Stationary_process)

<b>The convergence of Markov chain</b>

- (弱收敛条件) For any irreducible, finite, aperiodic Markov chain:
    1. All states are ergodic
    2. There exists a unique stationary distribution ${\pi}^{*}$
    3. Let $r_{ij}^{(t)}=P(X_{t}=j,\forall\ 1\leq{s}\leq{t-1},X_{s}\neq{j}|X_{0}=j)$, all states are persistent: 
    $$ \sum_{t=1}^{\infty}r_{ii}=1,\quad{}\sum_{t=1}^{\infty}r_{ii}t=\frac{1}{\pi_{i}}\qquad{}\forall\ i $$
    1. In $t$ steps, the chain visit state $i$ for $N(i,t)$ times:
   $$ \lim_{t\rightarrow{\infty}}\frac{N(i,t)}{t}=\pi_{i} $$

简单分析一下，先说这几个条件

- Irreducible表示不可约减，意思就是说这个无向图里面不能有“黑洞”这种随机游走时进得去出不来的子图。通常情况下连通图等价于不可约简
- Finite即该定理只适用于离散环境下
- Aperiodic Markov chain具有某种复杂性，不收敛到平稳分布。通常情况下判断aperiodic Markov chain最简单直接的方式是看一个图是不是二分图，如果是二分图的话这个图就是周期的
  
事实上大多数讲MCMC的博客都是上来直接讲detailed balance，MCMC中随机过程的构建也是是基于detailed balance的。但这个弱收敛定理也同样非常有意思，很多传统算法问题，包括PageRank、n-SAT问题等都可以随机化成离散Markov chain上的随机游走问题求解；强化学习中的MDP状态空间也常常是离散的，在一些比较简单的离散环境下我们可以只通过观察状态图是否满足上面这几个条件来判断是否存在唯一的平稳分布

## 4. Detailed balance

**Theorem:** Consider a finite, irreducible, aperiodic Markov chain with transition matrix $P$. If there are non-negative numbers $\pi=\{\pi_i\}$ such that $\sum_{i}π_{i}=1$, and for any pair of states $i, j$:
$$ \pi_{i}P_{i,j} = \pi_{j}P_{j,i} $$
Then π is the stationary distribution corresponding to $P$.

这个定理告诉我们，对于一个Markov chain而言，如果我们想证明其平稳分布存在，最常规的方法就是尝试证明对任意相邻状态$x$和$y$，都有$\pi_{x}P_{xy}=\pi_{y}P_{yx}$成立。~~没错这很数学，这个定理简直像是一句极其正确的废话~~

