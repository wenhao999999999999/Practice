// 动态规划解题套路框架
# include <bits/stdc++.h>
using namespace std;

// 首先，虽然动态规划的核心思想就是穷举求最值，但是问题可以千变万化，穷举所有可行解其实并不是一件容易的事，需要你熟练掌握递归思维，只有列出正确的「状态转移方程」，才能正确地穷举。
// 而且，你需要判断算法问题是否具备「最优子结构」，是否能够通过子问题的最值得到原问题的最值。
// 另外，动态规划问题存在「重叠子问题」，如果暴力穷举的话效率会很低，所以需要你使用「备忘录」或者「DP table」来优化穷举过程，避免不必要的计算。

// 动态规划的三要素：1.状态转移方程；2.最优子结构；3.重叠子问题
// 写状态转移方程思路：明确状态，也就是原问题和子问题中会变化的变量（一般是函数入参）-> 明确选择，也就是导致「状态」产生变化的行为 -> 明确dp函数的定义，返回值一般是题目要求我们计算的量。

// 动态规划算法本质上就是穷举「状态」，然后在「选择」中选择最优解。

class Solution {
public:
    // 1.零钱兑换
    // 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
    // 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
    vector<int> memo;

    int coinChange(vector<int>& coins, int amount) {
        // 备忘录，memo[i] 表示凑成总金额 i 做需要的最少硬币数目
       memo = vector<int> (amount+1, -666);
        return dp(coins, amount);
        
    }

    // dp函数定义：
    // 输入：硬币面额数组 coins， 总金额 amount;
    // 返回：凑成总金额所需的最少的硬币个数；
    int dp(vector<int>& coins, int amount) {
        // base case
        if (amount == 0) return 0;
        // 注意：添加总金额小于零判断
        if (amount < 0) return -1;

        if (memo[amount] != -666) return memo[amount];

        // 注意：res 要定义在 dp 函数体内部
        int res = INT_MAX;
        for (auto coin : coins) {
            // 计算子问题的答案
            int subproblem = dp(coins, amount - coin);
            
            // 注意：子问题无解则跳过
            if (subproblem == -1) continue;

            // 在子问题中寻找最优解，然后加一
            res = min(res, subproblem + 1);
        }
        // 注意：要判断是否有解
        memo[amount] = res == INT_MAX ? -1 : res;
        return memo[amount];
    }
};

// 2. 最长递增子序列
// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
// 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        // 数组的长度
        int n = nums.size();
        // dp 数组定义：dp[i] 表示以 nums[i] 这个数结尾的最长递增子序列的长度
        // 全部初始化为 1
        vector<int> dp(n, 1);

        // base case : dp 数组全部初始化为 1
        for (int i = 0; i < n; i++) {
            // nums[i] 前面的所有数都可以作为递增子序列的前缀
            // 注意：这里的 i 是从 0 开始的，所以 j 的范围是 [0, i-1]
            // 也就是 nums[i] 前面的所有数都可以作为递增子序
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = max(dp[i], dp[j]+1);
                }
            }
        }

        // 根据函数定义，返回值可以定义如下：
        int res = 1;
        for (int i = 0; i < dp.size(); i++) {
            res = max(res, dp[i]);
        }
        return res;
    }
 
};

// 3. 下降路径最小和
// 给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。
// 下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。

class Solution {
public:
    // 备忘录
    vector<vector<int>> memo;

    int minFallingPathSum(vector<vector<int>>& matrix) {
        int n = matrix.size();
        int res = INT_MAX;
        
        // 备忘录里的值初始化为 66666
        memo = vector<vector<int>>(n, vector<int>(n, 66666));

        // 终点可能回落到最后一行的任意一列
        for (int i = 0; i < n; i++) {
            res = min (res, dp(matrix, n-1, i));
        }

        return res;
    }

     // dp函数定义：从第一行（matrix[0][..]）向下落，落到位置 matrix[i][j] 的最小路径和为 dp(matrix, i, j)。
     // 输入：状态（落入的位置）
     // 返回：最小路径和
    int dp(vector<vector<int>>& matrix, int i, int j) {
        // 1.非法索引检查
        if (i < 0 || j < 0 ||
            j >= matrix.size() ||
            j >= matrix.size()) {
                return INT_MAX; // 因为我们调用的是 min 函数，最终返回的值是最小值，所以对于不合法的索引，只要 dp 函数返回一个永远不会被取到的最大值即可。
        }

        // 2.base case
        // 如何确定：回顾我们的 dp 函数定义：
        // 从第一行（matrix[0][..]）向下落，落到位置 matrix[i][j] 的最小路径和为 dp(matrix, i, j)。
        // 根据这个定义，我们就是从 matrix[0][j] 开始下落。那如果我们想落到的目的地就是 i == 0，所需的路径和当然就是 matrix[0][j] 呗。
        if (i == 0) {
            return matrix[0][j];
        }

        // 3.查找备忘录，防止重复计算
        if (memo[i][j] != 66666) {
            return memo[i][j];
        }

        // 状态转移
        memo[i][j] =  matrix[i][j] + min({
            dp(matrix, i-1, j),
            dp(matrix, i-1, j-1),
            dp(matrix, i-1, j+1) // i - 1, j - 1, j + 1 这几个运算可能会造成索引越界
        });

        return memo[i][j];
    }
};

// 4.单词拆分
// 给你一个字符串 s 和一个字符串数组 wordDict 作为字典，
// 请你判断是否可以利用字典中的单词拼接出 s 。
class Solution {
private:
    vector<string> wordDict;
    // 记录是否找到一个合法的答案
    bool found = false;

    // 记录回溯算法的路径
    vector<string> path;

    // 备忘录,存储不能切分的子串（子树），从而避免重复计算
    // 注意：这里使用 unordered_set 而不是 vector，因为我们只需要判断是否存在某个子串，而不需要知道它的具体位置
    // 这里的子串是 s 的一部分，所以我们使用 string 类型
    unordered_set<string> memo;
public:
    // 回溯算法
    bool wordBreak(string s, vector<string>& wordDict) {
        this->wordDict = wordDict;

        // 执行回溯算法，穷举所有可能的组合
        backtrack(s, 0);
        return found;
    }

    // 回溯算法框架
    void backtrack(string& s, int i) {
        // base case
        if (found) return; // 如果已经找到一个合法的答案，就不需要继续回溯了

        if (i == s.size()) {
            found = true; // 找到一个合法的答案
            return;
        }

        // 查询子串是否已经计算过
        string sub = s.substr(i);
        if (memo.count(sub)) {
            return; // 如果子串已经计算过，说明它不能切分，直接返回
        }

        // 回溯算法框架
        for (string& word : wordDict) {
            // 看看哪个单词能够匹配 s[i..i+len]
            int len = word.size();

            // 注意：这里的 i + len 不能超过 s.size()
            if (i+len <= s.size() && s.substr(i, len) == word) {
                // 找到一个单词匹配 s[i..i+len]
                // 做选择
                path.push_back(word);

                // 继续回溯,进入回溯树的下一层
                backtrack(s, i + len);

                // 撤销选择
                path.pop_back();
            }
        }

        // 后序位置，将不能切分的子串加入备忘录
        if (!found) {
            memo.insert(sub);
        }
    }
};

// 动态规划解法
class Solution {
private:
    unordered_set<string> wordDict;
    // 备忘录，-1 代表未计算，0 代表无法凑出，1 代表可以凑出
    vector<int> memo;
public:
    // 主函数
    bool wordBreak(string s, vector<string>& wordDict) {
        // 转化为哈希集合，快速判断元素是否存在
        for (auto &word : wordDict) {
            this->wordDict.insert(word);
        }

        // 初始化备忘录，-1 表示未计算
        memo.resize(s.length(), -1);

        return dp(s, 0);

    }

    // dp函数定义：返回 s[i..]是否能够被拼出
    int dp(string s, int i) {
        // base case
        if (i == s.size()) {
            return true; // 如果 i 到达字符串末尾，说明已经成功拼出
        }

        // 检查备忘录,防止冗余计算
        if (memo[i] != -1) {
            return memo[i] == 0 ? false : true; // 如果备忘录中已经计算过，直接返回结果
        }

        // 遍历字典中的单词
        for (auto &word : wordDict) {
            // 判断 s[i..i+len] 是否等于 word
            int len = word.size();
            if (s.substr(i, len) == word) {
                // 如果匹配成功，继续递归检查剩余部分
                if (dp(s, i + len)) {
                    memo[i] = 1; // 标记为可以拼出
                    return true; // 找到一个可以拼出的方案，直接返回
                }
            }
        }

        memo[i] = 0; // 标记为无法拼出
        return false; // 如果没有找到任何可以拼出的方案，标记为无法拼出
    }
};

// 4.2 单词拆分Ⅱ
// 给你一个字符串 s 和一个字符串数组 wordDict 作为字典，
// 请你将 s 中所有可以由字典中的单词组成的部分进行拆分，并返回所有可能的结果。
class Solution {
public:

};

// 5. 子序列类型问题
// 5.1 最大子数组和
// 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

// 滑动窗口解法：大于等于0时，扩大窗口；小于0时，缩小窗口
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        int left = 0;
        int right =0;
        int max_sum = INT_MIN;
        int cur_sum = 0;

        while (right < n) {
            // 扩大窗口
            cur_sum += nums[right];
            right++;

            // 更新答案
            max_sum = max(max_sum, cur_sum);

            // 缩小窗口
            while (cur_sum < 0) {
                cur_sum -= nums[left];
                left++;
            }
        }

        return max_sum;
    }
};

// 动态规划解法：以 nums[i] 为结尾的「最大子数组和」为 dp[i]，状态转移方程为：
// dp[i] = max(dp[i-1] + nums[i], nums[i])
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;

        // dp 数组定义：dp[i] 表示以 nums[i] 这个数结尾的最大子数组和
        vector<int> dp(n, 0);

        // base case
        dp[0] = nums[0];

        // 状态转移
        for (int i = 1; i < n; i++) {
            dp[i] = max(dp[i-1]+nums[i], nums[i]);
        }

        // 找到最大值
        int max_sum =  INT_MIN;
        for (int i = 0; i < n; i++) {
            max_sum = max(max_sum, dp[i]);
        }

        return max_sum;
    }
};

// 前缀和解法：
// 前缀和数组 pre_sum[i] 表示 nums[0..i-1] 的前缀和，即 pre_sum[i] = nums[0] + nums[1] + ... + nums[i-1]
// 则preSum[i+1] - preSum[j] 其实就是子数组 nums[j..i] 之和。那么反过来想，以 nums[i] 为结尾的最大子数组之和是多少？其实就是 preSum[i+1] - min(preSum[0..i])。
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;

        // 前缀和数组
        vector<int> pre_sum(n+1, 0);
        for (int i = 1; i <= n; i++) {
            pre_sum[i] = pre_sum[i-1] + nums[i-1];
        }

        // 最大子数组和
        int max_sum = INT_MIN;
        int min_pre_sum = INT_MAX;
        for (int i = 0; i < n; i++) {
            min_pre_sum = min(min_pre_sum, pre_sum[i]);
            max_sum = max(max_sum, pre_sum[i+1] - min_pre_sum);
        }

        return max_sum;
    }
};

// 5.2 最长公共子序列
// 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

// 动态规划解法：自顶向下
class Solution {
private:
    // 备忘录，消除重叠子问题
    vector<vector<int>> memo;

public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size();
        int n = text2.size();

        // 备忘录初始化值为-1，表示未计算
        memo = vector<vector<int>> (m, vector<int>(n, -1));
        return dp(text1, text2, 0, 0);
    }

    // 状态转移方程定义：计算 s1[i..] 和 s2[j..] 的最长公共子序列的长度
    int dp(string s1, string s2, int i, int j) {
        // base case
        if (i == s1.size() || j == s2.size()) {
            return 0;
        }

        // 避免重复计算
        if (memo[i][j] != -1) return memo[i][j];

        // 状态转移
        if (s1[i] == s2[j]) {
            // s1[i] 和 s2[j] 必然在公共子序列中，加s1[i+1..] 和 s2[j+1..]的最长公共子序列长度， 就是答案
            memo[i][j] = 1+dp(s1, s2, i+1, j+1);
        }else {
            // s1[i] 和 s2[j] 中至少有一个字符不在公共子序列中，穷举三种情况的结果，取其中对大的结果
            memo[i][j] = max (
                // 情况一：s1[i] 不在公共子序列中
                dp (s1, s2, i+1, j),
                // 情况二： s2[j] 不在公共子序列中
                dp (s1, s2, i, j+1)
            );
        }

        return memo[i][j];
    }
};

// 动态规划解法：自底向上
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size();
        int n = text2.size();

        // 定义二维数组 dp，dp[i][j] 表示 text1[0..i-1] 和 text2[0..j-1] 的最长公共子序列的长度
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        // 目标：text1[0..m-1] 和 text2[0..n-1] 的最长公共子序列的长度，即 dp[m][n]
        // base case: dp[0][..] = dp[..][0] = 0

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i-1] == text2[j-1]) {
                    dp[i][j] = 1 + dp[i-1][j-1];
                } else {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[m][n];
    }
};

// 爬楼梯
// 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
class Solution {
private:
    int res = 0;
    // 备忘录
    vector<int> memo;
public:
    int climbStairs(int n) {
        memo.resize(n+1, -1);
        res = dp(n);
        return res;
    }

    // dp 函数定义：输入需要爬的总楼梯阶数（状态）；确定选择：每次可以爬1阶或者2阶；返回值：爬到楼顶的方法的总数
    int dp (int n) {
        // base case
        if (n <= 0) return 0;
        if (n == 1) return 1; 
        if (n == 2) return 2;

        if (memo[n] != -1) return memo[n];

        // 如果选择爬1阶，有几种方法
        memo[n] = dp(n-1) + dp(n-2);

        return memo[n];
    }
};

// 最大子数组和
// 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。子数组是数组中的一个连续部分。
// 解题思路:dp函数返回以当前元素结尾的连续子数组的最大和
class Solution {
private:
    int maxSum = 0;
    int curSum = 0;
public:
    int maxSubArray(vector<int>& nums) {
        if (nums.size() == 1) return nums[0];
        // 注意:初始化要更新为第一个元素
        maxSum = nums[0];
        dp(nums, nums.size() - 1);
        return maxSum;
    }

    // dp 函数定义:确定状态,当前连续子数组和;确定选择,数组索引;返回值,以索引 i 结尾的连续子数组和的最大值;
    int dp (vector<int>& nums, int i) {
        // base case
        if (i == 0) return nums[i];

        // 注意:按照 dp 函数定义:curSum 是以 i 结尾的连续子数组和的最大值,并不是全局最大值
        curSum = max(nums[i], dp(nums, i-1) + nums[i]);
        // 注意:这里要更新全局最大值
        maxSum = max(curSum, maxSum);

        return curSum;
    }
};

// 打家劫舍(House Robber)
// 题干：
    // 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
// 思路:
    // 明确状态:面前房子的索引;
    // 明确选择:抢或不抢;
    // 返回值:从索引 i 开始偷窃,能获取的最高金额
class Solution {
private:
    vector<int> memo; // 备忘录，memo[i] 表示从第 i 间房子开始偷的最大金额
public:
    int rob(vector<int>& nums) {
        memo.resize(nums.size(), -1); // 初始化备忘录，-1 表示未计算
        return dp(nums, 0); // 从第 0 间房子开始偷
    }
    // 定义 dp(nums, start)：从索引 start 开始偷，能获得的最高金额。
    int dp (vector<int>& nums, int start) {
        // base case: 没有房子了
        if (start >= nums.size()) {
            return 0;
        }

        // 查备忘录，避免重复计算
        if (memo[start] != -1) return memo[start];

        // 选择：不偷当前，或者偷当前+跳过下一个
        int res = max(
                dp(nums, start+1),              // 不偷这家
                dp(nums, start+2) + nums[start] // 偷这家
            );

        memo[start] = res; // 保存结果到备忘录
        return memo[start];
    }
};


// 打家劫舍Ⅱ
// 题干：
    // 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
    // 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。
// 思路：
    // 把「环」拆成两条「直线」：
        // 情况一：考虑 nums[0..n-2]（偷第一间，不偷最后一间）。
        // 情况二：考虑 nums[1..n-1]（不偷第一间，可以偷最后一间）。
    // 两种情况分别套用「打家劫舍 I」的动态规划解法，取最大值。
// 代码：
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return nums[0]; // 特殊情况：只有一间房

        vector<int> memo1(n, -1);
        vector<int> memo2(n, -1);

        // 两次调用，分别对应两种情况
        return max(
            dp(nums, 0, n - 2, memo1), // 偷 [0..n-2]
            dp(nums, 1, n - 1, memo2)  // 偷 [1..n-1]
        );
    }

    // dp 函数：计算闭区间 [start, end] 的最优解
    int dp(vector<int>& nums, int start, int end, vector<int>& memo) {
        if (start > end) return 0;

        if (memo[start] != -1) return memo[start];

        // 状态转移：要么偷当前房子，要么跳过
        int res = max(
            dp(nums, start + 2, end, memo) + nums[start], // 偷当前
            dp(nums, start + 1, end, memo)                // 不偷
        );

        memo[start] = res;
        return res;
    }
};


// 买卖股票的最佳时机Ⅳ
// 给你一个整数数组 prices 和一个整数 k ，其中 prices[i] 是某支给定的股票在第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。也就是说，你最多可以买 k 次，卖 k 次。注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。我们把一次买入和一次卖出定义为一次「交易」

// 解题思路：动态规划解法，明确状态：天数，允许交易的最大次数，持有状态；明确选择：买入，卖出或者无操作，buy, sell, rest。 rest 操作还应该分两种状态，一种是 buy 之后的 rest（持有了股票），一种是 sell 之后的 rest（没有持有股票）；返回值：能获取的最大利润；
// 这个问题的「状态」有三个，第一个是天数，第二个是允许交易的最大次数，第三个是当前的持有状态（即之前说的 rest 的状态，我们不妨用 1 表示持有，0 表示没有持有）。然后我们用一个三维数组就可以装下这几种状态的全部组合：
// dp[i][k][0 or 1]
// 0 <= i <= n - 1, 1 <= k <= K
// n 为天数，大 K 为交易数的上限，0 和 1 代表是否持有股票。
// 此问题共 n × K × 2 种状态，全部穷举就能搞定。

// for 0 <= i < n:
//     for 1 <= k <= K:
//         for s in {0, 1}:
//             dp[i][k][s] = max(buy, sell, rest)
// class Solution {
// public:
//     int maxProfit(int k, vector<int>& prices) {
        
//     }
// };
// 我们想求的最终答案是 dp[n - 1][K][0]，即最后一天，最多允许 K 次交易，最多获得多少利润。

// 状态转移方程：
// dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
            //   max( 今天选择 rest,        今天选择 sell       )
// dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
//               max( 今天选择 rest,         今天选择 buy         )

// base case
// dp[-1][...][0] = 0
// 解释：因为 i 是从 0 开始的，所以 i = -1 意味着还没有开始，这时候的利润当然是 0。

// dp[-1][...][1] = -infinity
// 解释：还没开始的时候，是不可能持有股票的。
// 因为我们的算法要求一个最大值，所以初始值设为一个最小值，方便取最大值。

// dp[...][0][0] = 0
// 解释：因为 k 是从 1 开始的，所以 k = 0 意味着根本不允许交易，这时候利润当然是 0。

// dp[...][0][1] = -infinity
// 解释：不允许交易的情况下，是不可能持有股票的。
// 因为我们的算法要求一个最大值，所以初始值设为一个最小值，方便取最大值。

// 买卖股票的最佳时机
// 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

class Solution {
private:
    int count = 0;
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>>dp(n, vector<int>(2));

        for (int i = 0; i < n; i++) {
            if (i - 1 == -1) {
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }

            dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i]);
            // 注意：因为只有一次买入机会，所以如果选择今天买入的话，就表示之前都没有买入过，今天才第一次买入
            // dp[i-1][0] - prices[i] 的含义是：在之前已经卖出过（拿到了利润），然后今天再买入。
            dp[i][1] = max(dp[i-1][1], -prices[i]);
        }
        return dp[n-1][0];
    }
};

// 买卖股票的最佳时机Ⅱ
// 给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。返回 你能获得的 最大利润 。
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int dp_i_0 = 0;
        int dp_i_1 = INT_MIN;

        for (int i = 0; i < n; i++) {
            // 注意：这里要暂存前一天的状态值
            int temp = dp_i_0;
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = max(dp_i_1, temp - prices[i]);
        }

        return dp_i_0;
    }
};

// 买卖股票的最佳时机含冷冻期
// 给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格。设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
// 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
// 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）
// 解题思路：
// dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
// dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])
// 解释：第 i 天选择 buy 的时候，要从 i-2 的状态转移，而不是 i-1 。
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(2));

        for (int i = 0; i < n; i++) {
            if (i - 1 == -1) {
                // base case 1
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }

            if (i - 2 == -1) {
                // base case 2
                dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
                dp[i][1] = max(dp[i-1][1], -prices[i]);
                continue;
            }

            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i]);
        }

        return dp[n-1][0];
    }
};

// 买卖股票的最佳时机含手续费
// 给定一个整数数组 prices，其中 prices[i]表示第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。返回获得利润的最大值。
// 注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
// 解题思路：在每次卖出或者买入股票时减去手续费
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(2));

        for (int i = 0; i < n; i++) {
            if (i - 1 == -1) {
                // base case 1
                dp[i][0] = 0;
                dp[i][1] = -prices[i] - fee;
                continue;
            }

            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i] - fee);
        }

        return dp[n-1][0];
    }
};

// 买卖股票的最佳时机Ⅲ
// 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
// 解题思路：
// 原始的状态转移方程，没有可化简的地方
// dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
// dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int max_k = 2;
        // 注意：dp 是三维数组
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(max_k+1, vector<int>(2)));

        for (int i = 0; i < n; i++) {
            for (int k = max_k; k >= 1; k--) {
                // base case
                if (i - 1 == -1) {
                    dp[i][k][0] = 0;
                    dp[i][k][1] = -prices[i];
                    continue;
                }

                dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
                dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
            }

        }
        return dp[n-1][max_k][0];

    }
};