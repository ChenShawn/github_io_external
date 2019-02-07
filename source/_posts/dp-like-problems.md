---
title: DP类问题总结
date: 2019-01-05 23:12:34
tags: Coding
mathjax: true
---

*未完成，待更新。。。*
<!--more-->

## 494. Target Sum

> You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.
> 
> Find out how many ways to assign symbols to make sum of integers equal to target S.

暴力做法是$O(2^{N})$，不可取，转而研究这个题是否可以化为dp求解。算法导论里学过，如果一个问题可以用dp求解，那么它一定满足两个条件

- 存在最优子问题 optimal subproblem
- 子问题之间存在重叠 overlapping between subproblem

首先考虑一个trick来将这个问题化成一个类似于0-1背包的问题，设所有前面加正号的集合为$P$，负号的集合为$N$，那么有下面两个等式成立

$$P+N=Sum \\ P-N=target$$

可得$P=\frac{1}{2}(sum + target)$，至此就将题目转换成了一个类0-1背包问题——在数组中找一个子集$P$，使得P的和为$\frac{1}{2}(sum + target)$。容易想到，对于数组中的某个元素`nums[i]`，若数组存在若干个解可以加和得到S，那么也一定可以加和得到`S-nums[i]`。按照这个思路，开一个二维数组`dp[i][j]`表示前i个元素有`dp[i][j]`种方案可以加和得到`j`。接下来就非常简单了，复杂度为$O(NS)$，递推公式为
$$dp[i][j]=dp[i-1][j]+dp[i-1][j-nums[i]]$$

我的答案是直接照搬递推公式实现，但实际上空间复杂度还是可以继续优化的。假设数组`dp[i]`代表截至到下标i之前有多少种方案，由此只需要$dp[i]+=dp[i-nums[j]], \forall{j\in{nums}}, \forall{i\in{\{nums[j], nums[j]+1, ..., S\}}}$ （注意内循环要反过来，和0-1背包一样）

```cpp
class Solution {
public:
    int findTargetSumWays(const vector<int>& nums, int target) {
        int sum = 0;
        if (nums.empty())
            return 0;
        for (int val: nums) {
            sum += val;
        }
        if (sum < target || ((sum + target) & 1))
            return 0;
        else
            return findSubSet(nums, (sum + target) >> 1);
    }
    
    int findSubSet(const vector<int>& nums, int target) {
        vector<vector<int>> dp(nums.size(), vector<int>(target + 1, 0));
        dp[0][0] = 1;
        if (nums[0] <= target)
            dp[0][nums[0]] += 1;
        for (int i=1; i<nums.size(); i++) {
            for (int j=0; j<=target; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j - nums[i] >= 0)
                    dp[i][j] += dp[i - 1][j - nums[i]];
            }
        }
        return dp.back()[target];
    }
};
```

答案区里更简洁的代码

```cpp
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int s) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        return sum < s || (s + sum) & 1 ? 0 : subsetSum(nums, (s + sum) >> 1); 
    }   

    int subsetSum(vector<int>& nums, int s) {
        int dp[s + 1] = { 0 };
        dp[0] = 1;
        for (int n : nums)
            for (int i = s; i >= n; i--)
                dp[i] += dp[i - n];
        return dp[s];
    }
};
```

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

## 322. Coin Change

比较标准的完全背包题，开一个长度为amount + 1的数组dp，其中dp[i]代表凑够金额i最少需要使用dp[i]数量的硬币，由此可得递推公式为

$$dp[i]=\min{(dp[i], dp[i-v] + 1)} \quad{} \forall{v\in{coins}}$$

第一次写的时候写出的代码是三层循环，效率很低，如下

```cpp
const int MAXVAL = 100000001;
int coinChange(vector<int>& coins, int amount) {
    int tmp;
    vector<int> dp(amount + 1, MAXVAL);
    dp[0] = 0;
    for (int i=1; i<=amount; i++) {
        for (int val: coins) {
            for (int k=1; i - k * val >= 0; k++) {
                dp[i] = min(dp[i - k * val] + k, dp[i]);
            }
        }
    }
    return dp[amount] == MAXVAL? -1 : dp[amount];
}
```

事实上最里层的这个循环是不需要的，为什么，试考虑这个问题的最优子问题性质，可以想到，若`dp[i - val]`中存储了凑够金额`i - val`的最少硬币数量，那么`dp[i - val]`对于凑够金额`i - val`这个子问题是最优的，我们就无需再去搜索比`i - val`更小的子问题空间了。

最终24ms代码如下

```cpp
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, MAXVAL);
    dp[0] = 0;
    for (int i=1; i<=amount; i++) {
        for (int val: coins) {
            if (val <= i) {
                dp[i] = min(dp[i], dp[i - val] + 1);
            }
        }
    }
    return dp[amount] == MAXVAL? -1 : dp[amount];
}
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