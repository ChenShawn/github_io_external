---
title: DP类问题总结
date: 2019-01-05 23:12:34
tags: Coding
mathjax: true
---

*未完成，待更新。。。*
<!--more-->

## 123. Best Time to Buy and Sell Stock III

hard难度，最优解应该是$O(N)$，但我的$O(N^{2})$加了魔法优化trick之后居然歪打正着地跑到了6ms，击败了55.08%的submission。。。

平方级别复杂度的思路其实是很简单的，题目要求只能进行两次交易，且不能同时进行两笔交易，所以就循环找一个分割点，使得分割点左边的subarray单次交易+分割点右边的subarray单次交易最大即可。单次交易最大化收益的方法见[第121题](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)。

这种解法大数据量测试会超时，因此想出一个魔法trick来进行效率优化：考虑股价函数$f(t)$为一个光滑函数，若存在极大值或极小值的分割点$s$满足$a\leq{b}\leq{s}\leq{c}\leq{d}$，使得$f(b)-f(a)+f(d)-f(c)$最大。进一步可以想到$f''(s)>0$或$f''(s)<0$只需要满足二者一即可。因而可以在迭代分割点的时候，只对极大值点进行计算，代码如下

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.empty())
            return 0;
        int ans = 0, tmp = 0;
        for (int i=1; i<prices.size()-1; i++) {
            if (prices[i] > prices[i - 1] && prices[i] >= prices[i + 1]) {
                tmp = maxProfit(prices, 0, i + 1) + maxProfit(prices, i + 1, prices.size());
                if (tmp > ans)
                    ans = tmp;
            }
        }
        tmp = maxProfit(prices, 0, prices.size());
        return ans > tmp ? ans : tmp;
    }
    
    int maxProfit(vector<int>& prices, int begin, int end) {
        int ans = 0, tmp = 0, slow = begin, fast = begin;
        while (fast + 1 < end) {
            if (prices[fast + 1] >= prices[fast]) {
                tmp = prices[fast + 1] - prices[slow];
                if (tmp > ans)
                    ans = tmp;
                fast ++;
            } else {
                fast ++;
                if (prices[fast] < prices[slow])
                    slow = fast;
            }
        }
        return ans;
    }
};
```

## 64. Minimum Path Sum

> Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

很有意思的一道题，第一眼看题时候的反应是：欸这不是类似于value iteration嘛快赶紧写个DP出来展现一下某扎实的RL功底balabala。。。

然后就开始写value iteration了，说实话也是学了David Silver课程之后第一次写value iteration，代码如下

```cpp
class Solution {
    const int MAX_VAL = 0x7fffffff;
    
public:
    int minPathSum(vector<vector<int>>& grid) {
        if (grid.empty() || grid[0].empty())
            return 0;
        int m = grid.size(), n = grid[0].size(), comp;
        vector<vector<int>> dp(m, vector<int>(n, MAX_VAL));
        vector<vector<int>> dp_copy(m, vector<int>(n, MAX_VAL));
        dp[m - 1][n - 1] = grid[m - 1][n - 1];
        dp_copy[m - 1][n - 1] = grid[m - 1][n - 1];

        do {
            comp = dp[0][0];
            for (int i=m-1; i>=0; i--) {
                for (int j=n-1; j>=0; j--) {
                    if (i == m - 1 && j == n - 1)
                        continue;
                    dp_copy[i][j] = minAround(dp, i, j);
                    if (dp_copy[i][j] != MAX_VAL)
                        dp_copy[i][j] += grid[i][j];
                }
            }
            dp.assign(dp_copy.begin(), dp_copy.end());
        } while (comp == MAX_VAL || comp > dp[0][0]);
        return dp[0][0];
    }
    
    int minAround(const vector<vector<int>>& dp, int row, int col) {
        int up = row + 1 < dp.size() ? dp[row + 1][col] : MAX_VAL;
        int down = row - 1 >= 0 ? dp[row - 1][col] : MAX_VAL;
        int left = col - 1 >= 0 ? dp[row][col - 1] : MAX_VAL;
        int right = col + 1 < dp[0].size() ? dp[row][col + 1] : MAX_VAL;
        int lval = up < down ? up : down;
        int rval = left < right ? left : right;
        return lval < rval ? lval : rval;
    }
};
```

写value iteration有两个需要注意的点

- dp数组必须要有两个，dp_copy用来做更新，dp用来做旧的参照，每次迭代结束后要将dp_copy完整地复制给dp
- 迭代终止条件：按照Contractiom Mapping Theorem，value iteration每一步迭代必定会使得解空间可行域变小。对于这道题而言，`dp[0][0]`的值应该是（非严格）单调递减的。因而迭代终止条件可以表达为
    - 若`dp[0][0] == MAX_VAL`，则迭代一定还未结束。其原因在于这道题中，一定存在一条路径可以从左上到右下
    - 若`dp[0][0]`在某一次迭代中值变小，则迭代一定未结束
- 是否有更好的终止条件，现在暂时没有想出来

高高兴兴地写完提交，200+ms，足以说明这份代码的效率之低下。回头重新看题，发现漏掉了一个条件

> **Note**: You can only move either down or right at any point in time.

<div align="center">
    <img src="http://www.xfdown.com/uploads/allimg/1804/2-1P409151353232.png" width="200">
</div>

有了这个条件，这道题就会容易得多。事实上用value iteration来解这道题纯属杀鸡用牛刀。这里将value iteration的适用范围与此题做简单对比

- MDP的state transition是non-deterministic的
- value iteration主要解决的问题是faster convergence和loopy MDP

既然只能往下或往右走，那这道题就简化了很多，有DP和BFS求最短路两种解法，效率上应该是DP更优。一个显而易见的结论是每个`dp[i][j]`作为子问题也是最优的，那么递推公式应该为

```cpp
dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
```

最终的8ms代码如下

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        if (grid.empty() || grid[0].empty())
            return 0;
        int m = grid.size(), n = grid[0].size(), comp;
        vector<vector<int>> dp(m, vector<int>(n, 0));
        
        // boundary condition
        dp[0][0] = grid[0][0];
        for (int i=1; i<m; i++) 
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        for (int i=1; i<n; i++)
            dp[0][i] = dp[0][i -1] + grid[0][i];
        // transition function
        for (int i=1; i<m; i++) {
            for (int j=1; j<n; j++) {
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }
};
```

## 72. Edit Distance

经典DP问题，相比于之前的DP题，这个问题的最优子问题性质没有那么一目了然，直接放讲解与答案

> The idea would be to reduce the problem to simple ones. For example, there are two words, horse and ros and we want to compute an edit distance D for them. One could notice that it seems to be more simple for short words and so it would be logical to relate an edit distance D[n][m] with the lengths n and m of input words.
> 
> Let's go further and introduce an edit distance D[i][j] which is an edit distance between the first i characters of word1 and the first j characters of word2.
> 
> It turns out that one could compute D[i][j], knowing D[i - 1][j], D[i][j - 1] and D[i - 1][j - 1]
> 
> If the last character is the same, i.e. word1[i] = word2[j] then
> 
> $$dp[i][j]=1+\min(dp[i-1][j], dp[i][j-1],dp[i-1][j-1]-1)$$
> 
> and if not, i.e. word1[i] != word2[j] we have to take into account the replacement of the last character during the conversion.
> 
> $$dp[i][j]=1+\min(dp[i-1][j], dp[i][j-1],dp[i-1][j-1])$$

Runtime 12ms，time complexity $O(mn)$ where m is the length of s1 and n is the length of s2

```cpp
class Solution {
public:
    int minDistance(string& s1, string& s2) {
        if (s1.empty() && s2.empty())
            return 0;
        else if (s1.empty() || s2.empty())
            return abs(s1.size() - s2.size());
        vector<vector<int>> dp(s1.size() + 1, vector<int>(s2.size() + 1));
        for (int i=0; i<=s1.size(); i++)
            dp[i][0] = i;
        for (int i=0; i<=s2.size(); i++)
            dp[0][i] = i;
        
        for (int i=1; i<=s1.size(); i++) {
            for (int j=1; j<=s2.size(); j++) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]);
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] - 1) + 1;
                } else {
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]);
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[s1.size()][s2.size()];
    }
    
    constexpr static inline int abs(const int n) {
        return n > 0 ? n : -n;
    }
};
```

## 96. Unique Binary Search Trees

> Given n, how many structurally unique BST's (binary search trees) that store values 1 ... n?

这个题有一定的难度，虽说是DP，但不像其他的DP题有明显的套路痕迹。

这道题目的最优子问题是什么？可以想到，一颗有n个节点的树可能有的形状数量显然是和节点数量相关的，具体来说，如果$K_{i}$代表有i个节点的树有多少种形状，那么

$$K_{i}=\sum_{j=0}^{i-1}K_{j}K_{i-1-j}$$

再加上边界条件$K_{0}=1, K_{1}=1$，代码其实非常简短

```cpp
int numTrees(int n) {
    vector<int> dp(n + 1, 1);
    for (int i=2; i<=n; i++) {
        dp[i] = 0;
        for (int j=0; j<i; j++) {
            dp[i] += dp[i - 1 - j] * dp[j];
        }
    }
    return dp[n];
}
```

## 95. Unique Binary Search Trees II

在96题基础上的延伸，虽说是延伸，这道题目的DP套路同样不容易看出来。

刚看到这个题目的时候想到的一种解法是这样的

```cpp
vector<TreeNode*> generateTrees(int n) {
    if(n == 0)
        return vector<TreeNode*>(0);
    vector<int> perm(n);
    vector<TreeNode*> ans;
    for (int i=1; i<=n; i++) {
        perm[i - 1] = i;
    }
    do {
        TreeNode* root = generateTreeFromVector(perm);
        ans.push_back(root);
    } while (next_permutation(perm.begin(), perm.end()));
    return ans;
}

TreeNode* generateTreeFromVector(const vector<int>& perm) {
    TreeNode *root = new TreeNode(perm[0]);
    TreeNode **p = &root;
    for (int i=1; i<perm.size(); i++) {
        while ((*p) != NULL) {
            if (perm[i] < (*p)->val)
                p = &((*p)->left);
            else
                p = &((*p)->right);
        }
        *p = new TreeNode(perm[i]);
    }
    return root;
}
```

**但是这种解法是彻彻底底的错误**。如果用上一道题的递推公式验算，显然unique BST的数量和permutation的数量是不等的，例如给定数组[1, 3, 2]和[1, 2, 3]，二者构成的是同一颗BST。

正确的思路仍然沿用了上面那道题的思想，对于有n个节点的BST，若其根节点为i，那么其左子树就会有i-1个节点，右子树有n-i个节点。我们需要遍历所有1-n的数字为根的情况，并对其左子树和右子树进行递归。

出递归的边界条件为：当左子树或右子树的节点数量为0时，直接返回NULL

12ms代码如下

```cpp
class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        if(n == 0)
            return vector<TreeNode*>(0);
        vector<TreeNode*> ans;
        recursion(1, n, ans);
        return ans;
    }
    
    void recursion(int start, int end, vector<TreeNode*>& ans) {
        if (start > end) {
            ans.push_back(NULL);
            return;
        }
        for (int i=start; i<=end; i++) {
            vector<TreeNode*> left_ptrs, right_ptrs;
            recursion(start, i - 1, left_ptrs);
            recursion(i + 1, end, right_ptrs);
            for (TreeNode* left: left_ptrs) {
                for (TreeNode* right: right_ptrs) {
                    TreeNode *root = new TreeNode(i);
                    root->left = left;
                    root->right = right;
                    ans.push_back(root);
                }
            }
        }
    }
};
```
## 221. Maximal Square

> Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

这道题的难点在于

- 递推公式不是很显然，无法归纳到几类典型的DP问题中
- 一维DP的优化思路非常不直观

### 基础DP解法

定义$dp(i,j)$的值为从左上顶点到坐标$(i,\ j)$为止的最大正方形边长，则Solution中给出的递推公式是这样的

$$dp(i, j) = \min(dp(i-1, j), dp(i, j-1), dp(i-1, j-1)) + 1 \quad{} \text{if}\ matrix(i-1, j-1)==1$$

虽然LeetCode上给出了下面这张图片来便于理解，但是这个递推公式仍然非常不直观：一般如果DP所求解的最终目标是一个离散空间中的最大值的话，那么递推公式中也应当包含$\max$项，然而这个递推公式中是用$\min$来确保遍历中所到达的每个点都是正方形

除此以外，我们还需要在每一次更新时记录最大边长的值

<div align="center">
    <img src="https://leetcode.com/media/original_images/221_Maximal_Square.PNG?raw=true" width=550px>
</div>

代码如下

```cpp
int maximalSquare(vector<vector<char>>& matrix) {
    int m = matrix.size();
    if (m == 0)
        return 0;
    int n = matrix[0].size();
    int maxlen = 0;
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    for (int i=1; i<=m; i++) {
        for (int j=1; j<=n; j++) {
            if (matrix[i - 1][j - 1] == '1') {
                dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                maxlen = max(maxlen, dp[i][j]);
            }
        }
    }
    return maxlen * maxlen;
}
```

### 内存优化

这里为什么可以用内存优化策略？如果我们仔细观察基础DP解法的代码，会发现每次更新中，只用到了$dp(i-1,j),dp(i,j-1)$与$dp(i-1,j-1)$三个值，其中$dp(i,j-1)$是遍历中上一次所到达的坐标，因此只需要多开一个prev变量来记录之前的状态即可，而$dp(i-1,j-1)$可以用上一次dp更新前的数值来表示，因而也可以省去，唯一真正需要的是之前column的最大正方形边长。

因此，经优化后，可以用$dp(i)$表示到达第i列为止最大的正方形边长，代码如下

```cpp
int maximalSquare(vector<vector<char>>& matrix) {
    int m = matrix.size(), maxlen = 0, prev = 0, tmp;
    if (m == 0)
        return 0;
    int n = matrix[0].size();
    vector<int> dp(n + 1, 0);
    for (int i=1; i<=m; i++) {
        for (int j=1; j<=n; j++) {
            tmp = dp[j];
            if (matrix[i - 1][j - 1] == '1') {
                dp[j] = min(min(dp[j - 1], prev), dp[j]) + 1;
                maxlen = max(maxlen, dp[j]);
            } else {
                dp[j] = 0;
            }
            prev = tmp;
        }
    }
    return maxlen * maxlen;
}
```

## 673. Number of Longest Increasing Subsequence

> Given an unsorted array of integers, find the number of longest increasing subsequence.

- 数组`len[i]`表示到达下标i时，最长的LIS长度为`len[i]`
- 数组`cnt[i]`表示到达下标i时，有`cnt[i]`种长度为`len[i]`的LIS

```cpp
int findNumberOfLIS(vector<int>& nums) {
    int n = nums.size(), maxlen = 1, ans = 0;
    vector<int> cnt(n, 1), len(n, 1);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                if (len[j] + 1 > len[i]) {
                    len[i] = len[j] + 1;
                    cnt[i] = cnt[j];
                } else if (len[j] + 1 == len[i]) {
                    cnt[i] += cnt[j];
                }
            }
        }
        maxlen = max(maxlen, len[i]);
    }
    for (int i = 0; i < n; i++) 
        if (len[i] == maxlen)
            ans += cnt[i];
    return ans;
}
```

## 87. Cheapest Flights Within K Stops

> There are n cities connected by m flights. Each fight starts from city u and arrives at v with a price w.
> 
> Now given all the cities and flights, together with starting city src and the destination dst, your task is to find the cheapest price from src to dst with up to k stops. If there is no such route, output -1.

起初认为是BFS水题，结果BFS写了半天也没AC，实现代码如下

```cpp
class Solution {
    const int MAX_DIST = 0x7fffffff;
public:
    int findCheapestPrice(int n, vector<vector<int>>& edges, int src, int dst, int k) {
        queue<pair<int, int>> que;
        que.push(pair<int, int>(src, 0));
        vector<int> dist(n, MAX_DIST);
        dist[src] = 0;
        while (!que.empty()) {
            pair<int, int> from = que.front();
            if (from.second > k)
                break;
            que.pop();
            for (const vector<int>& e: edges) {
                if (e[0] == from.first) {
                    que.push(pair<int, int>(e[1], from.second + 1));
                    if (dist[from.first] != MAX_DIST && dist[from.first] + e[2] < dist[e[1]])
                        dist[e[1]] = dist[from.first] + e[2];
                }
            }
        }
        if (dist[dst] == MAX_DIST)
            return -1;
        else
            return dist[dst];
    }
};
```

问题出在哪里？问题在于距离更新的这一行

```cpp
if (dist[from.first] != MAX_DIST && dist[from.first] + e[2] < dist[e[1]])
    dist[e[1]] = dist[from.first] + e[2];
```

这里的更新每次都需要去直接更改dist，可以想到，输入变量edges的顺序会影响到更新后dist数组内的值。

举例说明，假设给定一个graph，src为A，dst点为C，且edges的顺序为
```
A-->B
B-->C
```
那么第一轮更新时会先更新原点A到B的距离，然后更新C的距离的时候B的距离又会影响到C的距离值。若edges的顺序倒置，那么第一轮更新就只会更新节点B的距离，不会更新C。由于更新中edges的顺序会影响到迭代次数，既然题意中要求了中转不能超过K次，那么代码中就无法对中转次数的变量进行跟踪。

事实上这个题目的标准解法居然还有一个响亮的名字，叫做Bellman Ford算法，详见[百度百科](https://baike.baidu.com/item/bellman-ford%E7%AE%97%E6%B3%95/1089090)或[Wikipedia](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)

思路和上面的基本相同，解决方法就是为dist数组储存一个完整的备份，每次迭代后将新迭代后的值赋值给备份即可（value iteration的既视感又出来了），代码如下

```cpp
class Solution {
    const int MAXVAL = 1e+8;
public:
    // Bellman Ford
    int findCheapestPrice(int n, vector<vector<int>>& edges, int src, int dst, int k) {
        vector<int> dist(n, MAXVAL);
        dist[src] = 0;
        vector<int> dist_copy(dist.begin(), dist.end());
        for (int i=0; i<=k; i++) {
            for (const vector<int>& e: edges) {
                dist_copy[e[1]] = min(dist_copy[e[1]], dist[e[0]] + e[2]);
            }
            dist.assign(dist_copy.begin(), dist_copy.end());
        }
        if (dist[dst] == MAXVAL)
            return -1;
        else
            return dist[dst];
    }
};
```