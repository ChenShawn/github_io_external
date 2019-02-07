---
title: git使用记录
date: 2019-01-18 17:39:03
tags: Manual
---

git一些不是很常用的功能经常忘记指令，在这里简单记录
<!--more-->
## 1. 冲突解决

通常发生在多人协作开发时，不同的人对同一分支做了不同的修改，在提交的时候会由于冲突而无法通过。一般如果远端分支比本地分支更新的话，会选择直接抛弃本地分支，将远端分支的内容全部覆盖到现有代码上

```bash
# 若本地存在未commit的改动，需要这条指令
$ git stash
$ git pull --rebase origin master
```

但如果本地分支的改动和远端分支的改动都想保留，或者希望自定义需要保留的内容的话，就需要解决冲突。首先我们知道 `git pull` 等同于 `git fetch` + `git merge`两条指令，解决冲突的步骤包括

```bash
# 下载远端分支到本地缓存区
$ git fetch
# 若本地分支的改动尚未保存，需要先将改动commit掉
$ git add .
$ git commit -m "xxxx"
$ git merge
# 执行git merge后会显示哪些文件可以自动merge，哪些文件存在冲突
# 自动merge的文件会显示auto merge，冲突的文件会显示大写的CONFICT
# 然后就需要定位到存在冲突的文件中解决冲突

# 解决完冲突后，下面两种方法都是可以的
# 方法1
$ git rebase --continue
# 方法2
$ git add .
$ git commit -m "xxxx"
```

## 2. 恢复本地缓冲区中暂存的分支

```bash
$ git stash
# xxxxxxxxxxxxxx
$ git stash pop
```