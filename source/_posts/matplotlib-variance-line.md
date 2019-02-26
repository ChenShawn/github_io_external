---
title: 使用matplotlib绘制带有方差区间的曲线
date: 2019-02-10 15:31:51
tags:
    - Machine learning
    - Data analysis
---

在强化学习的论文中经常可以看到一条收敛线，周围还有浅浅的范围线，那些范围线就是方差。那么如何用matplotlib来绘制带方差区间的收敛曲线呢？

<!-- more -->

## matplotlib.pyplot.fill_between

实现该绘图功能需要用到的最重要的一个就是`matplotlib.pyplot.fill_between`，参数说明详见[官方文档](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.fill_between.html#matplotlib.pyplot.fill_between)，函数原型如下

```python
matplotlib.pyplot.fill_between(x, 
                               y1, 
                               y2=0, 
                               where=None,
                               interpolate=False, 
                               step=None, *, 
                               data=None, **kwargs)
```

函数的功能是将两条曲线之间的面积用制定颜色填充，在绘图时我们只需要手动计算出方差区间，然后使用该函数填充区间即可

注意需要将alpha参数调小，从而降低填充区域的透明度，避免原来绘制的图像被填充区域覆盖掉

## Implementation

```python
# Suppose variable `reward_sum` is a list containing all the reward summary scalars
def plot_with_variance(reward_mean, reward_var, color='yellow', savefig_dir=None):
    """plot_with_variance
        reward_mean: typr list, containing all the means of reward summmary scalars collected during training
        reward_var: type list, containing all variance
        savefig_dir: if not None, this must be a str representing the directory to save the figure
    """
    lower = [x - y for x, y in zip(reward_mean, reward_var)]
    upper = [x + y for x, y in zip(reward_mean, reward_var)]
    plt.figure()
    xaxis = list(range(len(lower)))
    plt.plot(xaxis, reward_mean, color=color)
    plt.fill_between(xaxis, lower, upper, color=color, alpha=0.2)
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.title('The convergence of rewards')
    if savefig_dir is not None and type(savefig_dir) is str:
        plt.savefig(savefig_dir, format='svg')
    plt.show()
```