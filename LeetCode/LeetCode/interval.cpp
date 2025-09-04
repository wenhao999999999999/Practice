#include <bits/stdc++.h>
using namespace std;
// 区间问题练习

// 1.插入区间
// 题干：
    // 给你一个 无重叠的 ，按照区间起始端点排序的区间列表 intervals，其中 intervals[i] = [starti, endi] 表示第 i 个区间的开始和结束，并且 intervals 按照 starti 升序排列。同样给定一个区间 newInterval = [start, end] 表示另一个区间的开始和结束。
// 思路：
        // 步骤 1：处理不重叠的前置区间：遍历 intervals，将所有结束位置小于 newInterval 起始位置的区间直接加入结果；

        // 步骤 2：合并重叠区间：当遇到与 newInterval 重叠的区间，不断更新 newInterval 的左右边界：
            // 左边界取两者较小值；
            // 右边界取两者较大值；
            // 直到遍历完所有重叠区间，将最终合并后的 newInterval 加入结果。

        // 步骤 3：处理不重叠的后置区间：将剩余不重叠且在 newInterval 之后的区间，直接加入结果。
// 代码：
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        vector<vector<int>> res; // 存储最终结果
        int i = 0;                  // 遍历原区间的指针
        int n = intervals.size();    // 原区间数量

        // 步骤 1：添加新区间前的不重叠区间
        while (i < n && intervals[i][1] < newInterval[0]) {
            res.push_back(intervals[i]);
            i++;
        }

        // 步骤 2：合并所有重叠区间
        // 如果原区间的起始位置 <= 新区间的结束位置，说明两个区间有重叠
        while (i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = min(newInterval[0], intervals[i][0]); // 更新合并后的左边界
            newInterval[1] = max(newInterval[1], intervals[i][1]); // 更新合并后的右边界 
            i++;
        }

        res.push_back(newInterval);

        // 步骤 3：添加新区间后的不重叠区间
        while (i < n) {
            res.push_back(intervals[i]);
            i++;
        }

        return res;
    }
};

// 2. 汇总区间
// 题干：
    // 给定一个无重复元素的有序整数数组 nums 。区间 [a,b] 是从 a 到 b（包含）的所有整数的集合。
    // 返回恰好覆盖数组中所有数字的最小有序区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个区间但不属于 nums 的数字 x 。
    // 列表中的每个区间范围 [a,b] 应该按如下格式输出：
        // "a->b" ，如果 a != b
        // "a" ，如果 a == b
// 思路：
    // 

// 代码：
class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        vector<string> ans;
        int n = nums.size();
        for (int i = 0; i < n; ) {
            int start = nums[i];
            // 延伸连续段：使用 long long 防止 start+1 溢出
            while (i + 1 < n && (long long)nums[i + 1] == (long long)nums[i] + 1) {
                ++i;
            }
            // 现在 [start, nums[i]] 是一个完整区间
            if (start == nums[i]) ans.push_back(to_string(start));
            else ans.push_back(to_string(start) + "->" + to_string(nums[i]));
            ++i; // 开始下一个区间
        }
        return ans;
    }
};
