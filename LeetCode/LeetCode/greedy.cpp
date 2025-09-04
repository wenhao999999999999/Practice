#include <bits/stdc++.h>
using namespace std;

// 贪心算法解题框架
    // 贪心定义：
        // 就是说能够通过局部最优解直接推导出全局最优解。
    // 最优子结构的意思：
        // 现在我已经把所有子问题的最优解都求出来了，然后我可以基于这些子问题的最优解推导出原问题的最优解。
    // 贪心选择性质:
        // 我只需要进行一些局部最优的选择策略，就能直接知道哪个子问题的解是最优的了，且这个局部最优解可以推导出原问题的最优解。此时此刻我就能知道，不需要等到所有子问题的解算出来才知道。所以贪心算法的效率一般都比较高，因为它不需要遍历完整的解空间。

// 贪心算法解题步骤：
    // 贪心算法的关键在于问题是否具备贪心选择性质，所以只能具体问题具体分析，没办法抽象出一套固 定的算法模板或者思维模式，判断一道题是否是贪心算法。
    // 我的经验是，没必要刻意地识别一道题是否具备贪心选择性质。你只需时刻记住，算法的本质是穷举，遇到任何题目都要先想暴力穷举思路，穷举的过程中如果存在冗余计算，就用备忘录优化掉。
    // 如果提交结果还是超时，那就说明不需要穷举所有的解空间就能求出最优解，这种情况下肯定需要用到贪心算法。你可以仔细观察题目，是否可以提前排除掉一些不合理的选择，是否可以直接通过局部最优解推导全局最优解。

// 1.跳跃游戏:
// 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
    class Solution {
    public:
        // 解题思路：只要保证每一步都走的尽可能远 ，就是局部最优的
        bool canJump(vector<int>& nums) {
            int id = 0;
            for (int i = 0; i < nums.size(); i++) {
                // 注意：必须先判断是否卡在0这个位置了
                if (id < i) return false;

                // 注意：这里跟当前位置进行比较的是i + nums[i]， 表示在当前索引能到达的最远索引
                id = max(i + nums[i], id);
                if (id >= nums.size()-1) return true;
            }
            return true;
        }
    };


// 2.跳跃游戏Ⅱ：
// 给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。每个元素 nums[i] 表示从索引 i 向后跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:0 <= j <= nums[i]，i + j < n；返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。

// 解题思路：每次都选择最有潜力（覆盖范围最远的）局部最优解
    class Solution {
    public:
        int jump(vector<int>& nums) {
            // 增强鲁棒性
            if (nums.size() <= 1) return 0;

            // jumps 步可以跳到索引区间 [i, end]
            int end = 0;
            int jumps = 0;

            // 从 [i, end] 区间内起跳，可以跳到的最远索引
            int farthest = 0;
            // 注意：这里的判断条件是.size() - 1
            for (int i = 0; i < nums.size() - 1; i++) {
                // 从索引 i 能跳的最远的索引
                farthest = max(i + nums[i], farthest);

                if (i == end) {
                    // [i, end] 区间是 jumps 步可达的索引范围
                    // 现在已经遍历完 [i, end]，所以需要再跳一步
                    jumps++;
                    end = farthest;
                }

                if (end >= nums.size()-1) return jumps;

            }

            return -1;
        }
    };


// 3. 分发糖果
// 题干：
    // n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。
    // 你需要按照以下要求，给这些孩子分发糖果：
        // 每个孩子至少分配到 1 个糖果。
        // 相邻两个孩子中，评分更高的那个会获得更多的糖果。
        // 请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。
// 思路：
    // 1. 初始化：先给每个孩子 1 个糖果（保证基本条件）。
    // 2. 从左到右遍历：如果 ratings[i] > ratings[i-1]，那么 candies[i] = candies[i-1] + 1。（保证右边比左边分得多）
    // 3. 从右到左遍历：如果 ratings[i] > ratings[i+1]，那么 candies[i] = max(candies[i], candies[i+1] + 1)。（保证左边比右边分得多，同时保留之前可能已经加过的值）
    // 4. 最后把数组 candies 累加，得到最少糖果数。
// 代码:
    class Solution {
    public:
        int candy(vector<int>& ratings) {
            int n = ratings.size();
            vector<int> candies(n, 1);  // 每个孩子至少 1 个糖果

            // 从左到右
            for (int i = 1; i < n; i++) {
                if (ratings[i] > ratings[i - 1]) {
                    candies[i] = candies[i - 1] + 1;
                }
            }

            // 从右到左
            for (int i = n - 2; i >= 0; i--) {
                if (ratings[i] > ratings[i + 1]) {
                    // 注意点 1 :
                        // 在第二步（从右到左遍历）时，某个孩子可能已经在第一步（从左到右）时被分配过糖果。如果我们直接写 candies[i] = candies[i+1] + 1，就有可能 覆盖掉原来更大的值，从而破坏之前的条件。
                    candies[i] = max(candies[i], candies[i + 1] + 1);
                }
            }

            // 计算总和
            // 注意点 2 :
                // first：起始迭代器 last：结束迭代器（不包含） init：初始值（累加的起点）
            return accumulate(candies.begin(), candies.end(), 0);
        }
    };