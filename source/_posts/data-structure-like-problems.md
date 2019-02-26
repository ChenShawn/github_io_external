---
title: 数据结构类问题总结
date: 2019-01-10 21:45:45
tags: Coding
mathjax: true
---

## 基础：堆排序
<!-- more -->
算法分为两步

- **建堆**，对于所有的非叶子节点，从右下至左上依次调整节点顺序进行建堆
- **出堆**，对对中所有节点，从右下至左上，依次将每个节点与堆顶节点交换，从而使堆顶元素出堆，并调整堆中剩余元素使其仍为一个大顶堆/小顶堆

需要注意的几个点

- 大顶堆构造时比较操作符用大于号，即对于一个节点而言，若其左子节点或右子节点值大于父节点，则进行交换，小顶堆反之
- 构造大顶堆最终的排序结果是从小到大的，小顶堆反之
- 设堆中元素总数为$N$，若下标从0开始算，则非叶节点的节点范围是0到$\frac{N}{2}-1$；若下标从1开始，则非叶节点的节点范围是1到$\frac{N}{2}$

```cpp
void adjust(vector<int>& nums, int size, int root) {
    int left = ((root + 1) << 1) - 1, right = (root + 1) << 1;
    int maxidx = root;
    if (left < size && nums[left] > nums[maxidx])
        maxidx = left;
    if (right < size && nums[right] > nums[maxidx])
        maxidx = right;
    if (maxidx != root) {
        swap(nums[root], nums[maxidx]);
        adjust(nums, size, maxidx);
    }
}

void heap_sort(vector<int>& nums) {
    int size = nums.size();
    for (int i=size/2-1; i>=0; i--) {
        adjust(nums, size, i);
    }
    for (int i=size-1; i>=0; i--) {
        swap(nums[i], nums[0]);
        adjust(nums, i, 0);
    }
}
```

## 21. Merge Two Sorted Lists

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

## 23. Merge k Sorted Lists

> Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

21题升级版，hard难度，题意从merge两个链表变成merge k个排序过的链表。

### Approach 1: Compare one by one

最容易想到的方案就是21题的基础上简单变形，21题中我们沿着指针p遍历链表，直到找到第一个不小于指针q所指的值的节点，然后交换指针，继续遍历；这道题中，只需要把每一步中的比较操作变成遍历k个链表去比较即可。

```python
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        self.nodes = []
        head = point = ListNode(0)
        for l in lists:
            while l:
                self.nodes.append(l.val)
                l = l.next
        for x in sorted(self.nodes):
            point.next = ListNode(x)
            point = point.next
        return head.next
```

- **Time complexity**: $O(kN) where $N$ is the total number of nodes in the final returned linked-list.
- **Space complexity**: $O(N)$ creating a new linked-list with $N$ nodes.

### Approach 2: Optimize Approach 1 by Priority Queue

Almost the same as the one above but optimize the comparison process by priority queue. This approach can not only reduce the time complexity, but also significantly simplify the codes.

```cpp
struct Node_t {
    Node_t(ListNode* _node, int _val): node(_node), val(_val) {}
    bool operator > (const Node_t& cmp) const { return val > cmp.val; }
    ListNode *node;
    int val;
};

class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<Node_t, vector<Node_t>, greater<Node_t>> que;
        ListNode *head = new ListNode(0);
        ListNode *point = head;
        for (ListNode *ptr : lists) {
            if (ptr != NULL)
                que.push(Node_t(ptr, ptr->val));
        }
        while (!que.empty()) {
            Node_t tmp = que.top();
            que.pop();
            point->next = new ListNode(tmp.val);
            point = point->next;
            if (tmp.node->next != NULL) {
                que.push(Node_t(tmp.node->next, tmp.node->next->val));
            }
        }
        point = head->next;
        delete head;
        return point;
    }
};
```

- **Time complexity**: C++中的priority_queue底层是由大顶堆/小顶堆实现的，因此每次插入队列的复杂度为$O(\log(k))$，yielding a total time complexity of $O(N\log(k))$
- **Space complexity**: $O(N+k)$, where $O(N)$ for the new returned linked-list, and $O(k)$ for the priority_queue. In most cases, when $k$ is far less than $N$, the space complexity is $O(N)$.

### Other approaches

Refer to [the solution panel of this problem](https://leetcode.com/problems/merge-k-sorted-lists/solution/) for more approaches.

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

## 86. Partition List

> Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
> 
> You should preserve the original relative order of the nodes in each of the two partitions.

### 错误解法：链表快排

第一眼看到partition就想到链表快排，马上写出代码如下

```cpp
ListNode* partition(ListNode* head, int x) {
    ListNode *p = head, *q = head;
    while (p != NULL && q != NULL) {
        while (q != NULL && q->val < x)
            q = q->next;
        if (q != NULL) {
            for (p=q; p!=NULL && p->val >= x; )
                p = p->next;
            if (p != NULL)
                swap(p->val, q->val);
        }
    }
    return head;
}
```

但是这样的解法无法满足题意中第二个条件，即经partition后的两个子串节点顺序需要和之前一样，快速排序是不稳定的，显然无法达到这个要求。

### Two-pointers solution

另一种很简单的思路是构造两个链表`left`和`right`

- `left`链表包含所有值小于x的节点，`right`链表包含所有值大于等于x的节点
- 最后只需要将`left`链表的尾指针指向`right`链表的头指针，将`right`链表的尾指针置为NULL即可
- 具体实现时可以用一个指针`p`对原链表进行遍历，由于`left`与`right`链表的尾指针不会超过`p`，对尾指针的修改不会影响到指针`p`的下一步遍历，因此该方法的space complexity为$O(1)$

以下代码

```cpp
ListNode* partition(ListNode* head, int x) {
    ListNode left(0), right(0);
    ListNode *l = &left, *r = &right;
    while (head != NULL) {
        ListNode **ref = head->val < x? (&l) : (&r);
        (*ref)->next = head;
        (*ref) = (*ref)->next;
        head = head->next;
    }
    l->next = right.next;
    r->next = NULL;
    return left.next;
}
```

<b style="color:red;">Note</b>: 由于寻址`->`运算符优先级大于解引用`*`，上面代码中while循环里的括号不能省略，这道题虽然很简单，但是第一次提交时在这里debug花了非常多的时间