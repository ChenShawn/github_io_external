---
title: multiprocessing与threading模块相关踩坑记录
date: 2019-02-26 20:00:29
tags: 
    - Coding
    - Machine learning
---

## Background
<!-- more -->
多线程/多进程/分布式编程在深度学习/强化学习的应用中是很常见的问题，本文的问题就是在实现DPPO的时候遇到的。

在开始复现这个算法之前我已经参考运行了[Morvan大神的demo实现](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/DPPO.py)，这份代码的可读性非常棒，它的并行是按照读者-写者模式执行的，其中模型更新的master线程是读者，收集数据的worker线程是写者，双方的操作严格互斥：写者与环境交互得到训练数据放入队列，读者从队列中取出数据进行训练。后续测试中发现代码虽然确实实现了并行，然而运行效率并不是很高，运行时的CPU利用率始终保持在一个比较低的水平，经分析后原因主要有以下几点

- Python GIL的限制
- 对于读者而言，每次执行完一次更新后都会触发同步；对于写者而言，每次收集到一个batch的数据后也都会触发同步，因此很多时间会浪费在操作系统/线程级别的IO上
- 与A3C的实现不同，每次一个worker获取到数据时，不管其他worker处于什么状态，它们的buffer都会被清空，这就导致不管你开了多少个worker，最终只会有一个worker收集到的数据有效并传递给master线程。按照我个人的理解，这样实现的目的在于维持训练的稳定性——每当一个worker推送数据给master时，master都会进行模型参数的更新，而更新后的policy已经不是其他worker收集数据时的policy了，由于PPO方法只能用于on-policy，这部分数据理论上来讲应当舍弃

那么既然在异步调度中会有这么多限制，首先一个问题，是否可以抛弃一部分理论上的严谨性，把程序实现变成纯粹异步的？

答案是否定的，原因在于模型更新这一步无论如何都必须要进行同步，否则如果模型正在更新参数的时候worker运行，那么worker得到的trajectories就会是脏数据，因为这些trajectories从概率分布上讲既不服从旧的policy distribution，也不服从更新后的policy distribution。

那么进一步，是否可以只对于模型更新操作进行同步，剩余操作全部异步呢？

理论上来说似乎是可行的，然而后续的实验中发现，由于模型更新速度比worker收集trajectory快，大部分时间里master都会抢占掉锁，全局队列中的元素长期很少，这反而使得程序在操作系统/线程级别的IO上花费了更多的时间效率。

因此我最后选择了一个折中的方案，并将这份代码改成了自己的风格，这里总结下修改过的地方

- 设置一个队列大小的上限阈值`MAX_QSIZE`，同步操作仅发生在队列大小达到上限或队列为空时
  - 当队列大小达到上限，阻塞worker，进行模型更新直到队列为空
  - 当队列为空，阻塞master，启动所有worker异步收集数据
- 经验上来讲，PPO本就是TRPO的近似，而TRPO方法中每步更新的KL divergence upper bound是有理论保障的，因此每步更新policy distribution不会有太大变化，每次模型参数更新后可以不清空其他worker的buffer
- 考虑Python GIL的问题根深蒂固，用multiprocessing代替threading模块是更好的选择
- 子线程/进程的运行不阻塞主线程/进程，主线程实时进行evaluation和render
- 实例化一个Event类成员来管理训练的迭代停止，防止程序无法正常结束的情况

## multiprocessing

Python的multiprocessing库提供了与threading非常接近的API，且是由强变量类型的Python实现的，非常人性化，以下是几种使用multiprocessing创建进程的方式

#### 简单进程的创建

```python
import multiprocessing

def worker(num):
    """thread worker function"""
    print 'Worker:', num
    return

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()
```

#### 继承派生

```python
import multiprocessing

class Worker(multiprocessing.Process):
    def __init(self):
        super(Worker, self).__init__()

    def run(self):
        print 'In %s' % self.name
        return

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = Worker()
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
```

即使如此，如果你认为可以用与多线程并行的相同方式实现多进程并行，那将是调bug噩梦的开始。

<img src="http://img.99danji.com/uploadfile/2016/0419/20160419034745372.jpg" width="200px;">

所谓基础不牢地动山摇，如果你不明白其中原因，请重复仔细阅读下面这两句话：

> **线程是操作系统调度的最小单位，进程是操作系统中资源分配的最小单位**

换个说法

> **线程之间资源可以共享，进程则不然**

- 具体来说，如果每个子进程执行需要消耗的时间非常短，则不必使用多进程，因为进程的启动关闭也会耗费资源
- 使用多进程往往是用来处理CPU密集型的需求，如果是IO密集型则使用多线程去处理更加合适

### threading

## Our approach

## References

1. [Python multiprocess library documentary](https://docs.python.org/2/library/multiprocessing.html)
2. [一个讲的比较详细的博客](https://www.cnblogs.com/haitaoli/p/9837697.html)