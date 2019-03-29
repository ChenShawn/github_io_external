---
title: 凌乱的工作计划
date: 2019-01-01 18:08:15
tags: Personal
---

## On Exploration of Research Intern
<!-- more -->
- [MSRA: https://www.msra.cn/zh-cn/jobs](https://www.msra.cn/zh-cn/jobs)
- [Alibaba research intern: https://campus.alibaba.com/talentPlanDetail.htm](https://campus.alibaba.com/talentPlanDetail.htm?spm=a1z3e1.11770841.0.0&id=82)
- [腾讯实习生校园招聘](https://join.qq.com/index.php)
- [字节跳动: https://job.bytedance.com/intern](https://job.bytedance.com/intern)

## 状态汇总

*Note: 以下网页需要处于登录状态才可以正常查看*

- [Alibaba人才计划research intern](https://campus.alibaba.com/myJobApply.htm?spm=a1z3e1.11796652.0.0.7b5860d3YIJHPe)
- [腾讯深圳TEG](https://join.qq.com/center.php)
- [ByteDance AML Machine Learning Platform Intern](https://job.bytedance.com/user/profile/)

## TODO

*Note: first things first!*

- Devided-and-Conquer
- Re-implement GAIL-related works
  - AIRL，只是把D网络的结构改成了reward-shaping的形式，但有证明
  - 2017, Li *et al*, info-GAIL，解决multiple expert demonstration问题
  - ICLR-2019, Directed-info GAIL，由于涉及到一些option framework的内容，formulation还蛮复杂的
- Distributed DDPG的复现真是个悲伤的故事，即使只是在Pendulum-v0这样简单的任务上想复现出DDPG的效果也相当不容易（弃坑