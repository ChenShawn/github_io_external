---
title: 背包问题汇总
date: 2019-01-03 22:54:16
tags: Coding
mathjax: true
---

# Problem definition
<!-- more -->
## 0-1无价值

> Given n items with size A_i, an integer m denotes the size of a backpack. How full can you fill this backpack?

*注意：0-1背包问题中每件物品只能使用一次*

```cpp
int maxFilled(const vector<int>& sizes, int m) {
    vector<int> dp(m + 1, 0);
    for (int s: sizes) {
        for (int i=m; i>=s; i--) {
            dp[i] = max(dp[i - s] + s, dp[i]);
        }
    }
    return dp[m];
}
```

## 0-1有价值

> Given n items with size A_i and value V_i, and a backpack with size m. What's the maximum value you can put into the backpack?

```cpp
int maxValue(const vector<int>& sizes, const vector<int>& values, int m) {
    vector<int> dp(m + 1, 0);
    for (int i=0; i<sizes.size(); i++) {
        for (int j=m; j>=sizes[i]; j--) {
            dp[j] = max(dp[j - sizes[i]] + values[i], dp[j]);
        }
    }
    return dp[m];
}
```

## 完全背包

*和0-1背包唯一的区别在于物品件数变成无限个了*

*代码上的唯一区别是内循环的循环顺序，0-1背包是倒序遍历，完全背包是正序遍历*

> Given n kind of items with size A_i and value V_i (each item has an infinite number available) and a backpack with size m. What's the maximum value you can put into the backpack?

```cpp
int maxValue(const vector<int>& sizes, const vector<int>& values, int m) {
    vector<int> dp(m + 1, 0);
    for (int i=0; i<sizes.size(); i++) {
        for (int j=sizes[i]; j<=m; j++) {
            dp[j] = max(dp[j - sizes[i]] + values[i], dp[j]);
        }
    }
    return dp[m];
}
```

## 多重背包

> Given n items with size A_i, and a backpack of size m. Each item has an **finite** number N_i available and a corresponding value V_i. Find the maximum value you can put into the backpack.

*区别在于每样物品的数量是有限的*

- **方案一：** 如果物品的种类比较少，可以用多维数组来直接优化，[LeetCode474题Ones and Zeros](https://leetcode.com/problems/ones-and-zeroes/)是一个典型案例。

- **方案二：** 另一种方案是把问题转化成有$\sum_{i}N_{i}$件单独物品的0-1背包问题，三重循环

```cpp
int maxValue(vector<int>& sizes, vector<int>& nums, vector<int>& values, int m) {
    vector<int> dp(m + 1, 0);
    for (int i=0; i<sizes.size(); i++) {
        for (int k=0; k<nums[i]; k++) {
            for (int j=m; j>=sizes[i]; j--) {
                dp[j] = max(dp[j - sizes[i]] + values[i], dp[j]);
            }
        }
    }
    return dp[m];
}
```

- **方案三：** 
  - Intuition可以这样理解：如果某一类物品的个数为5，那么上面方案二就是相当于把这一类物品拆分成1+1+1+1+1=5，而方案三则相当于分解为4+1=5
  - 稍微严谨一点，给定整数$k$与一组$\alpha_{i}$满足$k=\sum_{i}\alpha_{i}2^{i}$，对于任何整数$0\leq{n}\leq{k}$，都存在一组整数$\beta_{i}\leq{\alpha_{i}}$使得$n=\sum_{i}\beta_{i}2^{i}$
  - 若某样物品的种类为$N_{i}$，这个定理可以使代码优化到$\sum_{i}O(\log(N_{i}))$

```cpp
int maxValue(vector<int>& sizes, vector<int>& nums, vector<int>& values, int m) {
    vector<int> dp(m + 1, 0);
    for (int i=0; i<sizes.size(); i++) {
        for (int k=1; k<=nums[i]; k*=2) {
            for (int j=m; j>=k * sizes[i]; j--) {
                dp[j] = max(dp[j - k * sizes[i]] + k * values[i], dp[j]);
            }
        }
    }
    return dp[m];
}
```

## 0-1背包方案数量

> Given n items with size A_i, and a backpack of size m. Find the number of possible ways to fill the backpack, return -1 if the filling is impossible.

```cpp
int numApproaches(const vector<int>& sizes, int m) {
    vector<int> dp(m + 1, 0);
    dp[0] = 1;
    for (int s: sizes) {
        for (int i=m; i>=s; i--) {
            dp[i] += dp[i - s];
        }
    }
    return dp[m];
}
```

## 完全背包方案数量

> Given n items with size A_i, and a backpack of size m. Each item has an infinite number available. Find the number of possible ways to fill the backpack, return -1 if the filling is impossible.

```cpp
int numApproaches(const vector<int>& sizes, int m) {
    vector<int> dp(m + 1, 0);
    dp[0] = 1;
    for (int s: sizes) {
        for (int i=s; i<=m; i++) {
            dp[i] += dp[i - s];
        }
    }
    return dp[m];
}
```

# LeetCode问题汇总

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

## 322. Coin Change

> You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

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

## 518. Coin Change 2

[Coin change](#322-Coin-Change)题的变形，依然是不限数目的硬币，求用现有的硬币种类凑够amount的方法有多少种

很容易想到递推式为$dp[i]=\sum_{v\in{\text{coins}}}dp[i-v]$，关键在于循环顺序，虽然这道题还是0-1背包，但如果还像之前一样循环的话就会出现一个问题：

比如说coins为[1,2]的一个集合，那么凑够amount=3的方法应该有两种，一种是1+1+1=3，另一种是会2+1=3，但是上面的循环方法会将1+2=3和2+1=3算成两种方法，导致输出不对

解决方案也很简单，把内外层循环调转位置即可

```cpp
int change(int amount, const vector<int>& coins) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    for (int val: coins) {
        for (int i=val; i<=amount; i++) {
            dp[i] += dp[i - val];
        }
    }
    return dp[amount];
}
```

## 416. Partition Equal Subset Sum

> Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

标准完全背包题，套模板硬做即可

```cpp
bool canPartition(vector<int>& nums) {
    int sum = 0;
    for (int n: nums)
        sum += n;
    if (sum & 1)
        return false;
    sum /= 2;
    vector<bool> dp(sum + 1, false);
    dp[0] = true;
    for (int val: nums) {
        for (int i=sum; i>=val; i--) {
            if (dp[i - val])
                dp[i] = true;
        }
    }
    return dp[sum];
}
```

## 474. Ones and Zeroes

> In the computer world, use restricted resource you have to generate maximum benefit is what we always want to pursue.
> 
> For now, suppose you are a dominator of m 0s and n 1s respectively. On the other hand, there is an array with strings consisting of only 0s and 1s.
> 
> Now your task is to find the maximum number of strings that you can form with given m 0s and n 1s. Each 0 and 1 can be used at most once.

```cpp
pair<int, int> count(const string& s) {
    pair<int, int> p(0, 0);
    for (char c: s) {
        if (c == '0')
            p.first ++;
        else
            p.second ++;
    }
    return p;
}

int findMaxForm(vector<string>& strs, int m, int n) {
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    for (const string& s: strs) {
        auto p = count(s);
        for (int i=m; i>=p.first; i--) {
            for (int j=n; j>=p.second; j--) {
                dp[i][j] = max(dp[i][j], dp[i - p.first][j - p.second] + 1);
            }
        }
    }
    return dp[m][n];
}
```