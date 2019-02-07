---
title: Sum类问题总结
date: 2019-01-03 13:48:50
tags: Coding
mathjax: true
---

*未完成，待继续更新。。。*

## 1. Two Sum

最基本的hash table题，复杂度$O(N)$
<!--more-->
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int, int> table;
        vector<int> ans;
        for (int i=0; i<nums.size(); i++) {
            table[nums[i]] = i;
        }
        for (int i=0; i<nums.size(); i++) {
            auto iter = table.find(target - nums[i]);
            if (iter != table.end() && iter->second != i) {
                ans.push_back(i);
                ans.push_back(iter->second);
                return ans;
            }
        }
        return ans;
    }
};
```

## 167. Two Sum II - Input array is sorted

一看到sorted就想到binary search，花了很大的时间代价思考，然而这题最优解是$O(N)$的，要用binary search反而得$O(N\log{N})$，得不偿失。

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> ans;
        int left = 1, right = nums.size();
        while (left < right) {
            if (nums[left - 1] + nums[right - 1] < target) {
                left ++;
            } else if (nums[left - 1] + nums[right - 1] > target) {
                right --;
            } else {
                ans.push_back(left);
                ans.push_back(right);
                return ans;
            }
        }
        return ans;
    }
};
```

有关binary search解法，唯一需要注意的点就是在求middle的时候，以往的不严谨写法一般是
```cpp
middle = (left + right) / 2;
```
其中`left + right`在数组较大时容易溢出，更加严谨的写法是
```cpp
middle = left + ((right - left) >> 1);
```

## 15. 3Sum

我人生中面试最丢人的时刻，就是在百度面试时被问及这道题的时候，那时我给出的解法是$O(N^{2}\log{N})$的，即先排序然后最内层循环用binary search搜。

然而这道题的解法意外的简单，假设三个数分别为a, b, c，思路就是固定a，然后用two sum的办法去找b和c，由于two sum有$O(N)$解，则此题复杂度为$O(N^{2})$

我的写法比较繁琐，耗时900+ms，说明这个写法思路虽然对了，但效率上有比较严重的问题。在这个case种，用iterator比直接用下标慢，且容易写错，在对iterator有更深入的理解之前应当避免使用这种写法。

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        if (nums.empty())
            return ans;
        int anchor;
        sort(nums.begin(), nums.end());
        auto it = nums.begin();
        while (it != nums.end()) {
            anchor = *it;
            vector<int> tmp(3, *it);
            twoSum(it, nums.end(), 0 - (*it), tmp, ans);

            // Skip duplicate elements
            while (it != nums.end() && (*it) == anchor)
                ++it;
        }
        return ans;
    }
    
    void twoSum(const vector<int>::iterator& begin, 
                const vector<int>::iterator& end,
                int target, vector<int>& tmp,
                vector<vector<int>>& ans) {
        vector<int>::iterator it = begin + 1;
        map<int, int> table;
        int anchor;
        while (it != end) {
            table[*it] = it - begin;
            ++it;
        }
        it = begin + 1;
        while (it != end) {
            anchor = *it;
            auto res = table.find(target - (*it));
            if (res != table.end() && res->second > (it - begin)) {
                tmp[1] = res->first;
                tmp[2] = *it;
                ans.push_back(tmp);
            }
            // Skip duplicate elements
            while (it != end && (*it) == anchor)
                ++it;
        }
    }
};
```

讨论区里最高赞答案如下，效率bottleneck主要在于map的开销，由于算法一开始已经将数组排好序，存map和查找的开销是可以省掉的。具体做法为

- 在搜2Sum的时候，用头尾两个指针`lo`和`hi`
- 若`nums[lo] + nums[hi] < target`，lo右移
- 若`nums[lo] + nums[hi] < target`，hi左移

```java
public List<List<Integer>> threeSum(int[] num) {
    Arrays.sort(num);
    List<List<Integer>> res = new LinkedList<>(); 
    for (int i = 0; i < num.length-2; i++) {
        if (i == 0 || (i > 0 && num[i] != num[i-1])) {
            int lo = i+1, hi = num.length-1, sum = 0 - num[i];
            while (lo < hi) {
                if (num[lo] + num[hi] == sum) {
                    res.add(Arrays.asList(num[i], num[lo], num[hi]));
                    // Skip duplicate elements
                    while (lo < hi && num[lo] == num[lo+1]) lo++;
                    while (lo < hi && num[hi] == num[hi-1]) hi--;
                    lo++; hi--;
                } else if (num[lo] + num[hi] < sum) {
                    lo++;
                } else {
                    hi--;
                }
           }
        }
    }
    return res;
}
```

## 18. 4Sum

既然3Sum只是在2Sum的基础上简单扩展，直接将这道题当做3Sum的扩展即可，复杂度$O(N^{3})$。代码写的比3Sum稍简洁了一些，去重的关键有两点

- 在for循环的末尾跳过之后的重复元素
- 一开始sort整个数组，并保证对每组答案`[a, b, c, d]`，必须满足`a <= b <= c <= d`

```cpp
// 28ms, faster than 61.58% of the codes 
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> ans;
        if (nums.empty())
            return ans;
        sort(nums.begin(), nums.end());
        
        for (int i=0; i<nums.size(); i++) {
            for (int j=i+1; j<nums.size(); j++) {
                twoSum(nums, target - nums[i] - nums[j], i, j, ans);
                while (j + 1 < nums.size() && nums[j] == nums[j + 1])
                    j++;
            }
            while (i + 1 < nums.size() && nums[i] == nums[i + 1])
                i++;
        }
        return ans;
    }
    
    void twoSum(vector<int>& nums, int target, int iidx, int jidx, vector<vector<int>>& ans) {
        int left = jidx + 1, right = nums.size() - 1;
        while (left < right) {
            if (nums[left] + nums[right] == target) {
                int tmp[4] = {nums[iidx], nums[jidx], nums[left], nums[right]};
                ans.push_back({tmp, tmp + 4});
                // Skip duplicate elements
                while (nums[left] == nums[left + 1])
                    left ++;
                while (nums[right] == nums[right - 1])
                    right --;
                left ++; right --;
            } else if (nums[left] + nums[right] < target) {
                left ++;
            } else {
                right --;
            }
        }
    }
};
```

## 112. Path Sum

二叉树，水题，只需要注意判断叶节点

```cpp
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if (root == NULL)
            return false;
        return hasPathSum(root, sum, 0);
    }
    
    bool hasPathSum(TreeNode* root, int sum, int depth) {
        if (root->left == NULL && root->right == NULL)
            return sum == root->val;
        if (root->left != NULL && hasPathSum(root->left, sum - root->val, depth + 1))
            return true;
        if (root->right != NULL && hasPathSum(root->right, sum - root->val, depth + 1))
            return true;
        return false;
    }
};
```

## 113. Path Sum II

虽然标了middle难度，但依然是水题，最基本的back-tracking操作

```cpp
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int>> ans;
        if (root == NULL) {
            return ans;
        }
        vector<int> tmp;
        pathSum(root, sum, tmp, ans);
        return ans;
    }
    
    void pathSum(TreeNode* root, int sum, vector<int>& tmp, vector<vector<int>>& ans) {
        if (root->left == NULL && root->right == NULL && sum == root->val) {
            tmp.push_back(root->val);
            ans.push_back(tmp);
            tmp.pop_back();
            return;
        }
        tmp.push_back(root->val);
        if (root->left != NULL)
            pathSum(root->left, sum - root->val, tmp, ans);
        if (root->right != NULL)
            pathSum(root->right, sum - root->val, tmp, ans);
        tmp.pop_back();
    }
};
```

## 494. Target Sum

题目是给一个数组，在每个数字前加正负号使得加和等于target的值，返回有多少种加和方案

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

## 653. Two Sum IV - Input is a BST

Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that their sum is equal to the given target.

核心思路两层循环，外循环遍历BST，内循环BST查找，复杂度$O(N\log{N})$

虽说也是水题，问题在于外循环因为每次要给内循环传根节点指针，所以不能用递归遍历，外层循环应该写成非递归——非递归的写法经常忘记，权当复习吧

```cpp
class Solution {
public:
    bool findTarget(TreeNode* root, int k) {
        if (root == NULL)
            return false;
        stack<TreeNode*> st;
        TreeNode *p = root, *q;
        while (p != NULL || !st.empty()) {
            while (p != NULL) {
                st.push(p);
                p = p->left;
            }
            if (!st.empty()) {
                p = st.top();
                q = findNode(root, k - p->val);
                if (q != NULL && q != p)
                    return true;
                st.pop();
                p = p->right;
            }
        }
        return false;
    }
    
    TreeNode* findNode(TreeNode* root, int k) {
        if (root == NULL)
            return NULL;
        if (root->val == k) {
            return root;
        } else if (root->val > k) {
            TreeNode *p = findNode(root->left, k);
            if (p != NULL)
                return p;
        } else {
            TreeNode *p = findNode(root->right, k);
            if (p != NULL)
                return p;
        }
        return NULL;
    }
};
```

## 39. Combination Sum

题意是给一个数组，找到这个数组中所有可以加和得到target的组合，每个数字可以使用无限次

很简单很基础的back-tracking题，复杂度为指数级别，然而我做了将近一个小时，动用了IDE来调bug才做出来，当前代码功力下降可见一斑

```cpp
class Solution {
public:
	vector<vector<int>> combinationSum(vector<int>& nums, int target) {
		vector<vector<int>> ans;
		if (nums.empty())
			return ans;
		sort(nums.begin(), nums.end());
		vector<int> tmp;
		for (int i = 0; i < nums.size(); i++) {
			dfs(nums, target, i, tmp, ans);
		}
		return ans;
	}

	void dfs(vector<int>& nums, int target, int begin, vector<int>& tmp, vector<vector<int>>& ans) {
		if (target < nums[begin])
			return;
		tmp.push_back(nums[begin]);
		if (target == nums[begin]) {
			ans.push_back(tmp);
			tmp.pop_back();
			return;
		}
		for (int i = begin; i<nums.size(); i++) {
			dfs(nums, target - nums[begin], i, tmp, ans);
		}
		tmp.pop_back();
	}
};
```

## 40. Combination Sum II

上面那道combinationSum的代码只把DFS迭代的起始位置从begin改成begin+1即可，水题，代码略

## 377. Combination Sum IV

这道题的返回值从返回所有的组合变成了返回有多少种组合方式，这显然是一道DP。开一个长度为target+1的数组dp，其中dp[i]代表有dp[i]种方式可以加和得到i，很容易看到递推公式应该是

$$dp[i]=\sum_{j=0}^{N}\mathbb{I}(nums[j]<=i)dp[i-nums[j]]$$

C++测试0ms代码如下

```cpp
int combinationSum4(vector<int>& nums, int target) {
    if (nums.empty())
        return 0;
    sort(nums.begin(), nums.end());
    vector<int> dp(target + 1, 0);
    dp[0] = 1;
    for (int i=1; i<=target; i++) {
        for (int val: nums) {
            if (val > i)
                break;
            else
                dp[i] += dp[i - val];
        }
    }
    return dp[target];
}
```