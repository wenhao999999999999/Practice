#include <bits/stdc++.h>
using namespace std;
// 双指针问题

// 1.判断子序列
// 题干：
    // 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

    // 进阶：如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？

// 思路：
    // 定义两个指针i,j，分别指向两个字符串s, t，遍历字符串 t ,指针 j 不断右移，判断两指针所指字符是否相等。
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int i = 0;
        int j = 0;

        while (j < t.size()) {
            if (s[i] == t[j]) {
                i++;
            }
            j++;
        }

        return i == s.size();
    }
};

// 2. 最长回文子串
// 题干：
    // 给你一个字符串 s，找到 s 中最长的回文子串。
// 思路：
    // 寻找回文串的问题核心思想是：从中间开始向两边扩散来判断回文串。
    // 找回文串的关键技巧是传入两个指针 l 和 r 向两边扩散，因为这样实现可以同时处理回文串长度为奇数和偶数的情况。
// 代码：
class Solution {
public:
    string longestPalindrome(string s) {
        string res = "";
        for (int i = 0; i < s.length(); i++) {
            // 以 s[i] 为中心的最长回文子串
            string s1 = palindrome(s, i, i);
            // 以 s[i] 和 s[i+1] 为中心的最长回文子串
            string s2 = palindrome(s, i, i + 1);
            // res = longest(res, s1, s2)
            res = res.length() > s1.length() ? res : s1;
            res = res.length() > s2.length() ? res : s2;
        }
        return res;
    }

private:
    string palindrome(string s, int l, int r) {
        // 防止索引越界
        while (l >= 0 && r < s.length() && s[l] == s[r]) {
            // 向两边展开
            l--;
            r++;
        }
        // 此时 [l+1, r-1] 就是最长回文串
        // 注意点1：while 循环是先判断左右指针所指字符是否相等再移动指针，因此在计算起点和长度的时候要先把指针移动到实际位置。
        return s.substr(l + 1, r - l - 1);
    }
};

// 三数之和问题
// 题干：
    // 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
// 思路：
    // nSum 系列问题的核心思路就是排序 + 双指针。 
    // 先给数组从小到大排序，然后双指针 lo 和 hi 分别在数组开头和结尾，这样就可以控制 nums[lo] 和 nums[hi] 这两数之和的大小：
    // 如果你想让它俩的和大一些，就让 lo++，如果你想让它俩的和小一些，就让 hi--。
    // 基于两数之和可以得到一个万能函数 nSumTarget，扩展出 n 数之和的解法，具体分析见详细题解。
// 代码：
class Solution {
public:
    /**
     * @brief  求解三数之和问题 (LeetCode 15. 3Sum)
     * 
     * 给定一个整数数组 nums，返回所有和为 0 且不重复的三元组。
     * 内部调用通用的 nSumTarget 函数。
     * 
     * @param nums 输入数组，会在函数内部排序
     * @return vector<vector<int>> 所有不重复的三元组，按任意顺序返回
     */
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        return nSumTarget(nums, 3, 0, 0);
    }

private:
    /**
     * @brief  通用 nSum 求解函数
     * 
     * 在有序数组 nums 中，寻找从索引 start 开始的 n 个数，
     * 使它们的和等于 target，返回所有不重复的 n 元组。
     * 
     * 算法思想：
     * - 当 n==2 时，退化为 2Sum 双指针求解；
     * - 当 n>2 时，枚举一个固定数，递归求解 (n-1)Sum。
     * 
     * @param nums   已经排序的数组
     * @param n      要寻找的数的个数 (n >= 2)
     * @param start  搜索起始下标，避免重复使用前面的元素
     * @param target 目标和
     * @return vector<vector<int>> 所有满足条件的不重复 n 元组
     */
    vector<vector<int>> nSumTarget(vector<int>& nums, int n, int start, long long target) {
        int sz = nums.size();
        vector<vector<int>> res;

        // 如果 n < 2（无意义）或数组剩余元素不足以凑出 n 个，直接返回空结果。
        if (n < 2 || sz < n) return res;

        if (n == 2) { // base case：n == 2（2Sum 双指针）
            int lo = start, hi = sz - 1;
            while (lo < hi) {
                long long sum = 1LL * nums[lo] + nums[hi];
                int left = nums[lo], right = nums[hi];
                if (sum < target) {
                    while (lo < hi && nums[lo] == left) lo++; // 和小了：左指针右移，并跳过相等值
                } else if (sum > target) {
                    while (lo < hi && nums[hi] == right) hi--; // 和大了：右指针左移，并跳过相等值  
                } else {
                    res.push_back({left, right}); // 命中一个解
                    while (lo < hi && nums[lo] == left) lo++; // 跳过与 left 相同的值，避免重复解
                    while (lo < hi && nums[hi] == right) hi--; // 同理，跳过与 right 相同的值
                }
            }
        } else { // 递归情况：n > 2
            for (int i = start; i < sz; ++i) {
                auto sub = nSumTarget(nums, n - 1, i + 1, target - nums[i]); // 枚举当前位置 i 作为固定的一个数 nums[i]。
                for (auto& arr : sub) {
                    arr.push_back(nums[i]);
                    res.push_back(arr);
                }
                while (i < sz - 1 && nums[i] == nums[i + 1]) ++i; // 去重
            }
        }
        return res;
    }
};

