---
title: 数据结构类问题总结
date: 2019-01-10 21:45:45
tags: Coding
mathjax: true
---

## 21. Merge Two Sorted Lists
<!--more-->
很简单的题，用了二级指针，4 lines 12ms。

```cpp
ListNode* mergeTwoLists(ListNode* a, ListNode* b) {
    ListNode **p = &a, **q = &b;
    while ((*q) != NULL) {
        while (*p != NULL && (*p)->val <= (*q)->val)
            p = &((*p)->next);
        swap(*p, *q);
    }
    return a;
}
```

## 530. Minimum Absolute Difference in BST

easy难度的题，已经是第二次做这道题了，但是还是花了很长时间才AC。

```cpp
class Solution {
public:
    int getMinimumDifference(TreeNode* root) {
        if (root == NULL)
            return 0;
        return midOrder(root);
    }
    
    int midOrder(TreeNode *root) {
        stack<TreeNode*> st;
        TreeNode *p = root;
        int cur = 1000000, last = -1000000, ans = 10000000;
        while (p != NULL || !st.empty()) {
            while (p != NULL) {
                st.push(p);
                p = p->left;
            }
            if (!st.empty()) {
                p = st.top();
                st.pop();
                
                cur = p->val;
                if (cur - last < ans)
                    ans = cur - last;
                last = p->val;
                
                p = p->right;
            }
        }
        return ans;
    }
};
```

## 222. Count Complete Tree Nodes

> Given a complete binary tree, count the number of nodes.

这道题的重点在于如何把完全二叉树的性质利用起来，第一次做没有做出来，[讨论区中C++的答案](https://leetcode.com/problems/count-complete-tree-nodes/discuss/61953/Easy-short-c%2B%2B-recursive-solution)技巧性非常强，先上代码

```cpp
int countNodes(TreeNode* root) {
    if (root == NULL)
        return 0;
    int left = 0, right = 0;
    TreeNode *p = root, *q = root;
    while (p != NULL) {
        left += 1;
        p = p->left;
    }
    while (q != NULL) {
        right += 1;
        q = q->right;
    }
    if (left == right)
        return (1 << left) - 1;
    else
        return countNodes(root->left) + countNodes(root->right) + 1;
}
```

第一眼看上去这份答案的复杂度仍然还是$O(N)$，还比普通的递归数节点多了判断左子树与右子树的操作，但是仔细考虑则不然，给定一颗高度为$h$的完全二叉树，对于每一个节点，可以归纳出以下3种情况

- a) 左子树与右子树的绝对高度都是$h$
- b) 左子树高度为$h$，右子树高度为$h-1$
- c) 左子树与右子树的高度都是$h-1$

可以想到对于情况a与c，这个结点下的子树都是满二叉树，因而可以直接用公式$2^{d}-1$算出节点数量，只有情况b需要靠递归来计数节点数量

那么情况b有多少节点呢？可以想到整颗完全二叉树中，这样的节点数量在$O(\log(N))$量级，i.e., 从根结点到叶节点，每层只会有一个节点满足条件b。因此，该算法的复杂度为$O(\log(n)\times{\log(N)})$