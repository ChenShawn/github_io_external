---
title: 奇技淫巧类题目总结
date: 2019-01-06 19:30:31
tags: Coding
mathjax: true
---

*未完成，待更新。。。*
<!--more-->

# Binary Search

## 4. Median of Two Sorted Arrays

> There are two **sorted** arrays nums1 and nums2 of size m and n respectively.
> 
> Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
> 
> You may assume A and B cannot be both empty.

头条最终面被问过的问题，显然是binary search，但是当时就是没想出来怎么写

Consider split indices i and j in array A and B correspondingly, visualizing as follows.

| left_part | right_part |
| ------ | ------ |
| $A_{0},...,A_{i-1}$ | $A_{i},...,A_{m-1}$ |
| $B_{0},...,B_{j-1}$ | $B_{j},...,b_{n-1}$ |

Provided that `len(left_part) == len(right_part)` and `min(A[i], B[j]) >= max(A[i - 1], B[j - 1])`, the split indices i and j must satisfy

$$
	i+j=\lceil{\frac{1}{2}(m+n)}\rceil=\frac{1}{2}(m+n+1) \\
	\Rightarrow j=\frac{1}{2}(m+n+1)-i
$$

复杂度$O(\log(\min(A,B)))$，这里的min是通过A与B指针交换实现的。看懂了上面的公式，这道题已经完成了80%，但这并不意味着剩下的20%也同样一目了然。剩余的20%与edge case相关，共有这么几类

- `i == 0 or j == 0`: Since A and B can not be both empty
	- if `i == 0`, find `B[j - 1]`
	- if `j == 0`, find `A[i - 1]`
- `i == m or j == n`: 
	- if `i == m`, find `B[j]`
	- if `j == n`, find `A[i]`
- `(m + n) % 2`: The definition of median is different for array of even or odd lengths. Specifically, for this problem
	- if `(m + n) % 2 == 1`: median is `max(A[i-1], B[j-1])`
	- if `(m + n) % 2 == 0`: median is `(max_of_left + min_of_right)`

```python
def findMedianSortedArrays(self, A, B):
	"""
	:type nums1: List[int]
	:type nums2: List[int]
	:rtype: float
	"""
	m, n = len(A), len(B)
	if m > n:
		A, B, m, n = B, A, n, m

	imin, imax, half_len = 0, m, (m + n + 1) / 2
	while imin <= imax:
		i = (imin + imax) / 2
		j = half_len - i
		if i < m and B[j-1] > A[i]:
			# i is too small, must increase it
			imin = i + 1
		elif i > 0 and A[i-1] > B[j]:
			# i is too big, must decrease it
			imax = i - 1
		else:
			# i is perfect
			if i == 0: max_of_left = B[j-1]
			elif j == 0: max_of_left = A[i-1]
			else: max_of_left = max(A[i-1], B[j-1])

			if (m + n) % 2 == 1:
				return max_of_left

			if i == m: min_of_right = B[j]
			elif j == n: min_of_right = A[i]
			else: min_of_right = min(A[i], B[j])

			return (max_of_left + min_of_right) / 2.0
```

# Back-tracking

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

## 22. Generate Parentheses

> Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

热身难度的back-tracking题，但是想把代码写到比较简洁，需要把思路理顺。这道题的解法并不唯一，这次我的答案是将解空间看成一颗二叉树来求解的

何解？将每种string的状态看成是二叉树的一个节点，对于每个节点而言，若其不是叶子节点，那么这个节点就会有两种分支情况

- [左子树] 若加入一个左括号合法，则加入一个左括号
- [右子树] 若加入一个右括号合法，则加入一个右括号
- [叶子节点] 若左括号与右括号都无法加入，开始回溯

那么这道题就转换成了一个遍历二叉树叶子节点的问题，代码如下

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        string tmp;
        generateParenthesis(n, 0, ans, tmp);
        return ans;
    }
    
    void generateParenthesis(int nleft, int nright, vector<string>& ans, string& s) {
        if (nleft == 0 && nright == 0) {
            ans.push_back(s);
            return;
        }
        if (nleft != 0) {
            s.push_back('(');
            generateParenthesis(nleft - 1, nright + 1, ans, s);
            s.pop_back();
        }
        if (nright != 0) {
            s.push_back(')');
            generateParenthesis(nleft, nright - 1, ans, s);
            s.pop_back();
        }
    }
};
```

## 51. N-Queens

经典back-tracking题。处理起来的麻烦之处在于每次back-tracking结束时要把queen占掉的位置还原回去，这一步琐碎且耗时。做这道题的时候干脆每一层递归都将棋盘标志位board变量复制一遍，以一些不易优化的效率换来了代码的干净整洁

```cpp
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> ans;
        vector<string> tmp(n, string(n, '.'));
        vector<vector<bool>> board(n, vector<bool>(n, true));
        solveNQueens(n, 0, board, tmp, ans);
        return ans;
    }
    
    void solveNQueens(int n, int row, vector<vector<bool>>& board, 
                      vector<string>& tmp, vector<vector<string>>& ans) {
        if (row == n) {
            ans.push_back(tmp);
            return;
        }
        for (int i=0; i<n; i++) {
            if (board[row][i]) {
                vector<vector<bool>> board_copy(board.begin(), board.end());
                setBoard(board_copy, n, row, i);
                tmp[row][i] = 'Q';
                solveNQueens(n, row + 1, board_copy, tmp, ans);
                tmp[row][i] = '.';
            }
        }
    }
    
    void setBoard(vector<vector<bool>>& board, int n, int row, int col) {
        for (int i=0; i<n; i++)
            board[row][i] == false;
        for (int i=0; i<n; i++)
            board[i][col] = false;
        int r = row, c = col;
        while (r < n && c < n)
            board[r++][c++] = false;
        r = row, c = col;
        while (r >= 0 && c >= 0)
            board[r--][c--] = false;
        r = row, c = col;
        while (r < n && c >= 0)
            board[r++][c--] = false;
        r = row, c = col;
        while (r >=0 && c < n)
            board[r--][c++] = false;
    }
};
```

## 52. N-Queens II

水题，上面的代码稍作改动即可，代码略

## 5. Longest Palindromic Substring

水题，犯了低级错误没能秒

```cpp
class Solution {
public:
    string longestPalindrome(const string& s) {
        if (s.size() <= 1)
            return s;
        int odd = 0, even = 0, ans = 0, axis;
        for (int i=0; i<s.size()-1; i++) {
            // Palindrome length is odd
            odd = expandPalindrome(s, i, i);
            // Palindrome length is even
            even = expandPalindrome(s, i, i + 1);
            if (odd > ans) {
                ans = odd;
                axis = i;
            }
            if (even > ans) {
                ans = even;
                axis = i;
            }
        }
        if (ans & 1) {
            return s.substr(axis - (ans >> 1), ans);
        } else {
            return s.substr(axis - (ans >> 1) + 1, ans);
        }
    }
    
    int expandPalindrome(const string& s, int axis, int raxis) {
        int left = axis, right = raxis;
        while (left >=0 && right < s.size() && s[left] == s[right]) {
            left --; right ++;
        }
        return right - left - 1;
    }
};
```

## 219. Contains Duplicate II

Easy难度的题，题意如下

> Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

最简单直接的做法是每次迭代开一个大小为k的滑动窗口来查找，复杂度$O(kN)$

```cpp
bool containsNearbyDuplicate(vector<int>& nums, int k) {
    if (nums.empty())
        return false;
    for (int i=0; i<nums.size(); i++) {
        for (int j=i+1; j<nums.size() && j-i<=k; j++) {
            if (nums[j] == nums[i])
                return true;
        }
    }
    return false;
}
```

但上面的代码运行时间2000+ms，这种思路必然是非常低效的，可以想到每次迭代都会有很多重复的查找。

稍微好一点的思路是对每个大小为k的滑动窗口维持一个hash表来查找，假设哈希表增删改查复杂度都是常数级别，则这种做法的复杂度为$O(N)$

```cpp
bool containsNearbyDuplicate(vector<int>& nums, int k) {
    if (nums.empty())
        return false;
    if (k >= nums.size())
        k = nums.size() - 1;
    set<int> table(nums.begin(), nums.begin() + k + 1);
    if (table.size() != k + 1)
        return true;
    for (int i=k+1; i<nums.size(); i++) {
        table.erase(nums[i - k - 1]);
        if (table.find(nums[i]) != table.end())
            return true;
        table.insert(nums[i]);
    }
    return false;
}
```

## 220. Contains Duplicate III

> Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.

题意要比之前的更加tricky，隐藏了若干edge case。

- t的值可以为负
- nums数组中两个元素相减有可能会超出int的范围

做这个题的的时候因为要用到set容器红黑树里上界与下界的查找，专门去查了[C++ reference的网站](http://www.cplusplus.com/reference/set/set/lower_bound/)，发现set容器没有查找正好比元素小的节点的API，后发现`set<T>::iterator`是有`--`operator的，可以直接用`set<T>::lower_bound`函数返回的iterator--来拿到所需结果。

```cpp
class Solution {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, long long k, long long t) {
        if (nums.empty() || t < 0)
            return false;
        if (k >= nums.size())
            k = nums.size() - 1;
        set<long long> table(nums.begin(), nums.begin() + k + 1);
        if (table.size() != k + 1)
            return true;
        for (auto it=table.begin(); it!=table.end(); it++) {
            auto cmp = it;
            cmp ++;
            if (cmp != table.end() && *cmp - *it <= t)
                return true;
        }
        for (int i=k+1; i<nums.size(); i++) {
            table.erase(nums[i - k - 1]);
            auto it = table.lower_bound(nums[i]), jt = it;
            if (it != table.end() && *it - nums[i] <= t)
                return true;
            if (jt != table.begin()) {
                jt --;
                if (nums[i] - *jt <= t)
                    return true;
            }
            table.insert(nums[i]);
        }
        return false;
    }
};
```

## 11. Container With Most Water

很有趣的一道题，思路很巧妙，代码意外地非常简单。

> Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
> 
> Note: You may not slant the container and n is at least 2.

<img src="https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg" width="700">

```cpp
int maxArea(vector<int>& height) {
    int maxarea = 0, left = 0, right = height.size() - 1;
    while (left < right) {
        maxarea = max(maxarea, min(height[left], height[right]) * (right - left));
        if (height[left] < height[right])
            left ++;
        else
            right --;
    }
    return maxarea;
}
```

> How this approach works?
> 
> Initially we consider the area constituting the exterior most lines. Now, to maximize the area, we need to consider the area between the lines of larger lengths. <b style="color:red">If we try to move the pointer at the longer line inwards, we won't gain any increase in area, since it is limited by the shorter line.</b> But moving the shorter line's pointer could turn out to be beneficial, as per the same argument, despite the reduction in the width. This is done since a relatively longer line obtained by moving the shorter line's pointer might overcome the reduction in area caused by the width reduction.
> 
> For further clarification [click here](https://leetcode.com/problems/container-with-most-water/discuss/6099/yet-another-way-to-see-what-happens-in-the-on-algorithm) and for the proof [click here](https://leetcode.com/problems/container-with-most-water/discuss/6089/anyone-who-has-a-on-algorithm).