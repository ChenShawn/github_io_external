---
title: CUDNN_STATUS_NOT_INITIALIZED
date: 2019-01-11 15:26:22
tags: Manual
---

## 错误信息
<!--more-->
服务器之前的tensorflow环境莫名其妙地坏掉了，定义graph的时候没有问题，一旦代码运行到`sess.run()`，就会触发报错

> UnknownError (see above for traceback): Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.

另一种报错信息是这样的

> 2019-01-11 15:54:13.351947: E tensorflow/stream_executor/cuda/cuda_dnn.cc:373] Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED
2019-01-11 15:54:13.352012: E tensorflow/stream_executor/cuda/cuda_dnn.cc:381] Possibly insufficient driver version: 384.130.0
2019-01-11 15:54:13.352024: E tensorflow/stream_executor/cuda/cuda_dnn.cc:373] Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED
2019-01-11 15:54:13.352042: E tensorflow/stream_executor/cuda/cuda_dnn.cc:381] Possibly insufficient driver version: 384.130.0
2019-01-11 15:54:13.352049: F tensorflow/core/kernels/conv_grad_input_ops.cc:981] Check failed: stream->parent()->GetConvolveBackwardDataAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(stream->parent()), &algorithms)
Aborted (core dumped)

## 环境:

- Linux Ubuntu 16.04.3
- tensorflow 1.12.0
- CUDA 9.1
- cudnn 7.0

## 解决方案

StackOverFlow+Github+CSDN搜索得到的解决方案经测试全部无效，记录如下

- StackOverFlow上很多人说这是由于GPU上已经有进程导致的，因而要在代码中加入下面的代码来限制GPU占用，经测试无效
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run()
    ...
```
- 针对上面第二种报错，CSDN上有人说是cudnn版本太低而导致的，遂重装cudnn，换新版本，经测试无效
- 使用sudo权限运行代码，即使用`sudo python baseline.py`而不是`python baseline.py`。经测试，这种方法不会触发上述报错信息，但程序运行到`sess.run()`的时候会卡住并停止响应，必须手动kill才能真正的结束进程
- tensorflow版本太高导致，强制pip重装低版本tensorflow，原博客给出的命令为
```bash
sudo pip3 install --upgrade --force-reinstall tensorflow-gpu==1.9.0 --user
```
经测试无效，且这样下载的tensorflow-gpu版本会是python 2.7的，这显然不合理，因此将其改为以下命令，后经测试无效
```bash
pip install tensorflow-gpu==1.9.0
```

## <b style="color: red;">最终解决问题的方案</b>

仔细研究了一下报错信息，似乎是说NVIDIA驱动的版本太低，和cudnn的版本不匹配，因而想到更新NVIDIA驱动。

安装或更新NVIDIA驱动的文章网上极多，这里列出几个查资料的时候读过的

- [[专业亲测]Ubuntu16.04安装Nvidia显卡驱动（cuda）--解决你的所有困惑](https://blog.csdn.net/ghw15221836342/article/details/79571559)
- [Ubuntu下安装nvidia显卡驱动（安装方式简单）](https://blog.csdn.net/linhai1028/article/details/79445722)
- [Ubuntu 16.04安装NVIDIA驱动](https://blog.csdn.net/u014797226/article/details/79626693)

这些文章的内容大同小异，总结下来有以下几个步骤

- 开始安装驱动之前要把Xorg或nouveau一类会占用显卡资源的进程全部关掉
- 用bash或者chmod一类的方法执行.run文件，后面加`–no-opengl-files`参数
- `sudo reboot`

安装结束后发现`nvidia-smi`无法正常显示，报错信息为

> Failed to initialize NVML: Driver/library version mismatch.

抓耳挠腮，只得再卸载nvidia驱动相关的一些组建，参考博客

- [解决Driver/library version mismatch](https://comzyh.com/blog/archives/967/)
- [NVIDIA驱动问题解决方案：Failed to initialize NVML: driver/library version mismatch](https://my.oschina.net/wangsifangyuan/blog/1606093)
- [**最有用的一篇博客**](https://blog.csdn.net/jiandanjinxin/article/details/80688900)

在命令行中执行

```bash
$ sudo rmmod nvidia_uvm
$ sudo rmmod nvidia_modeset
$ sudo rmmod nvidia
```

如果执行过程中报错说有其他项目对这几个存在依赖，则先行rmmod掉存在依赖的项目。执行结束后再`nvidia-smi`，发现驱动的版本也成功从384.0升级到了390.25，且之前的cudnn问题也误打误撞地解决了

<div align="center">
    <img src="http://pic.downyi.com/upload/2018-8/2018815183334229320.jpg" width="500" />
</div>