---
title: hexo使用记录
tags: Manual
# mathjax: true
---

## 1. 安装nodejs

[nodejs官网windows版本下载链接](https://nodejs.org/en/)，一路傻瓜式安装，记得把环境变量加上

## 2. 安装hexo
<!--more-->
- 先创建一个文件夹（用来存放所有blog），然后cd到该文件夹下
- 安装hexo: `npm i -g hexo`
- 初始化hexo: `hexo init`

## 3. 创建github.io

- 在自己的github账号下创建一个repo，名字叫`yourname.github.io`，其中yourname=github用户名
- 如果没有配置过这个github账号的话，要先在本地配置github账号信息
    ``` bash
    git config --global user.name "yourname"
    git config --global user.email "your_email@youremail.com"
    ssh-keygen -t rsa -C “your_email@youremail.com”
    ```
    然后会要求输入用户名密码，成功的话会在~/下生成.ssh文件夹，cd进去，`cat id_rsa.pub`，复制里面的key
    
    打开[github账号account settings下的ssh keys](https://github.com/settings/keys)，点new SSH key，title随便填，Key把刚才的内容黏贴进去

## 4. 修改_config.yml配置

### 1. 通过hexo deploy部署
用编辑器打开blog项目，修改_config.yml文件的配置 <span style="color:red;"><b> (注意冒号之后都是有一个半角空格!) </b></span>，做好这一步之后就可以直接通过`hexo d`免密码部署到自己的github主页了

```
deploy:
    type: git
    repo: https://github.com/YourgithubName/YourgithubName.github.io.git
    branch: master
```

### 2. post加密

- 运行 `npm install hexo-encrypt --save`
- 在_config.yml中添加变量
    ```
    encrypt: 
        password: UserDefinedPassWord
    ```
- 通过 `hexo new post "post_name"` 生成的markdown文件最前面都会有变量栏，形如：
    ```
    ---
    title: xxx
    date: yy-mm-dd hour:min:sec
    tags: xxx
    ---
    ```
    在需要加密的markdown文件最前面加 `encrypt: true` 即可

### 3. 修改hexo主题

网上讲解这个的博客很多，比如 [https://www.jianshu.com/p/33bc0a0a6e90](https://www.jianshu.com/p/33bc0a0a6e90)

### 4. TEX语法支持

作者在配置TEX语法支持功能的时候参考了若干篇博客，链接如下
- [https://www.jianshu.com/p/68e6f82d88b7](https://www.jianshu.com/p/68e6f82d88b7) 
- [https://blog.csdn.net/crazy_scott/article/details/79293576](https://blog.csdn.net/crazy_scott/article/details/79293576)
- [https://blog.csdn.net/u014792304/article/details/78687859](https://blog.csdn.net/u014792304/article/details/78687859)
  
Trouble shooting:
- 不能用默认的landscape主题，landscape对mathjax的支持很差
- 使用第三方主题时，注意第三方主题中的`_config.yml`文件中`mathjax`选项的配置
    - mathjax: true # 在第三方theme的_config.yml中配置
    - mathjax: enable # 在根目录下的_config.yml中配置
    - per_page: false # false的时候需要在markdown中手动添加mathjax: true来启动mathjax

问题测试样例（如果看到这个公式显示异常的话就需要调整.yml文件）:
$$\eta_*^\lambda=\sum_{i=0}^{T}D_{t_{e}^{s}}tk$$

### 5. 添加pdf文件

- 在博客bash中执行
    ``` bash
    npm install --save hexo-pdf
    ```
- 在markdown中需要用到pdf文件的地方加上
    ``` html
    {% pdf http://some_url.com/some_pdf.pdf %}
    ```
- 在blog的source/_post路径下新建一个文件夹，<span style="color:red"><b>名字与引用pdf的markdown文件名相同</b></span>，放入要上传的pdf文件


## 5. Hello world HEXO

Welcome to [Hexo](https://hexo.io/)! This is your very first post. Check [documentation](https://hexo.io/docs/) for more info. If you get any problems when using Hexo, you can find the answer in [troubleshooting](https://hexo.io/docs/troubleshooting.html) or you can ask me on [GitHub](https://github.com/hexojs/hexo/issues).

### 1. Create a new post

``` bash
$ hexo new "My New Post"
```

More info: [Writing](https://hexo.io/docs/writing.html)

### 2. Run server

``` bash
$ hexo server
```

More info: [Server](https://hexo.io/docs/server.html)

### 3. Generate static files

``` bash
$ hexo generate
```

More info: [Generating](https://hexo.io/docs/generating.html)

### 4. Deploy to remote sites

``` bash
$ hexo deploy
```

More info: [Deployment](https://hexo.io/docs/deployment.html)