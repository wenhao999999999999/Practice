// 回溯算法
# include <bits/stdc++.h>
using namespace std;

// 回溯/ DFS 算法解题框架：
// DFS/回溯算法的本质就是递归遍历一棵穷举树（多叉树），而多叉树的递归遍历又是从二叉树的递归遍历衍生出来的。所以我说 DFS/回溯算法的本质是二叉树的递归遍历。

// 1. 全排列（元素不重，不可复选）
// 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
class Solution {
    // 存储结果
    vector<vector<int>> res;

    // 存储当前路径
    vector<int> path;

    // 备忘录:记录已经访问过的元素
    vector<bool> visited;
public:
    vector<vector<int>> permute(vector<int> &nums) {
        // 初始化备忘录
        visited.resize(nums.size(), 0);
        backtracking(nums);
        return res;
    }

    // 回溯函数
    void backtracking(vector<int> &nums) {
         
        // base case
        if (path.size() == nums.size()) {
            res.push_back(path);
        }

        for (int i = 0; i < nums.size(); i++) {
            // 判断当前元素是否已经在路径中
            if (visited[i]) continue; 

            // 1.做选择
            path.push_back(nums[i]);
            visited[i] = 1;

            // 2.递归
            backtracking(nums);

            // 3.撤销选择
            path.pop_back();
            visited[i] = 0;
        }
          
    }
};

// 全排列Ⅱ（元素可重不可复选）
// 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
// 解题思路：保证相同元素在排列中的相对位置保持不变。
class Solution {
public:
    // 存储结果
    vector<vector<int>> res;

    // 存储当前路径
    vector<int> path;

    // 备忘录:记录已经访问过的元素
    vector<bool> visited;

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        // 排序，为剪支做准备
        sort(nums.begin(), nums.end());

        // 注意：初始化visited 数组
        visited.resize(nums.size(), 0);

        backtracking(nums);

        return res;
    }

    // 回溯函数
    void backtracking(vector<int> &nums) {
         
        // base case
        if (path.size() == nums.size()) {
            res.push_back(path);
        }

        for (int i = 0; i < nums.size(); i++) {
            // 判断当前元素是否已经在路径中
            if (visited[i]) continue; 

            // 注意：固定相同元素的使用顺序，只有相同元素的前一个元素使用过了，才能使用当前元素
            if (i > 0 && nums[i-1] == nums[i] && !visited[i-1]) continue;

            // 1.做选择
            path.push_back(nums[i]);
            visited[i] = 1;

            // 2.递归
            backtracking(nums);

            // 3.撤销选择
            path.pop_back();
            visited[i] = 0;
        }
          
    }    
};


// 2.岛屿问题
// 解题思路：用 DFS 方法遍历二维数组中的元素；如果发现陆地，结果+1；将与该陆地相连的区域淹没
class Solution {
public:
    // 从 (i, j) 开始，将与之相邻的陆地都变成海水
    void dfs(vector<vector<char>>& grid, int i, int j) {
        int m = grid.size();
        int n = grid[0].size();

        // 边界条件：如果越界或已经访问过，直接返回
        if (i < 0 || j < 0 || i >= m || j >= n) return;

        
        // 如果当前位置是水，直接返回
        if (grid[i][j] == '0') return;

        // 前序位置：将 (i, j) 变成海水
        grid[i][j] = '0';

        // 淹没上下左右的陆地
        dfs(grid, i-1, j); // 上
        dfs(grid, i+1, j); // 下
        dfs(grid, i, j-1); // 左
        dfs(grid, i, j+1); // 右

    }

    // 岛屿数量
    int numIslands(vector<vector<char>>& grid) {
        // 获取矩阵的行数和列数
        int m = grid.size();
        int n = grid[0].size();

        // 存储结果
        int res = 0;

        // 遍历 grid
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // 如果当前位置是陆地且未访问过
                if (grid[i][j] == '1') {
                    // 增加岛屿数量
                    res++;

                    // 然后使用 DFS 将岛屿淹了
                    dfs(grid, i, j);
                }
            }
        }
        return res;
    }
};

// 3.封闭岛屿的数量
// 给你一个 m x n 的二进制矩阵 grid ，其中 0 代表陆地，1 代表水域。
// 岛屿是由四面相连的陆地单元格（即上下左右）组成的封闭区域。
// 封闭岛屿是完全被水域包围的岛屿（即岛屿的所有边界都被水域包围）。
// 请你返回 封闭岛屿 的 数目 。
class Solution {
public:
    int closedIsland(vector<vector<int>>& grid) {
        // 获取矩阵的行数和列数
        int m = grid.size();
        int n = grid[0].size();

        // 先把边界上的陆地都淹没掉
        for (int i = 0; i < m; i++) {
            // 注意这边要用递归把所有和边界相连的陆地都淹没掉
            dp(grid, i, 0); // 左边界
            dp(grid, i, n-1); // 右边界
        }

        for (int j = 0; j < n; j++) {
            dp(grid, 0, j); // 上边界
            dp(grid, m-1, j); // 下边界
        }

        int res = 0;

        // 遍历grid,剩下的岛屿都是封闭岛屿
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    // 如果当前位置是陆地且未访问过
                    res++; // 增加封闭岛屿数量

                    // 深度优先搜索，淹没所有相连的陆地
                    dp(grid, i, j);
                }
            }
        }

        return res;
    }

    // 从 (i, j) 开始，将与之相邻的陆地都变成海水
    void dp(vector<vector<int>>& grid, int i, int j) {
        // 非法索引检查
        if (i < 0 || j < 0 ||
            i >= grid.size() || j >= grid[0].size()) {
                return;
        }

        // 如果当前位置是水域，直接返回
        if (grid[i][j] == 1) {
            return;
        }

        // 将当前位置变成水域
        grid[i][j] = 1;

        // 淹没上下左右的陆地
        dp(grid,i-1, j); // 上
        dp(grid,i+1, j); // 下
        dp(grid,i, j-1); // 左
        dp(grid,i, j+1); // 右

    }
};

// 4.岛屿的最大面积
// 给你一个 m x n 的二进制矩阵 grid ，其中 0 代表水域，1 代表陆地。
// 岛屿是由四面相连的陆地单元格（即上下左右）组成的封闭区域。
// 请你计算并返回该岛屿的最大面积。如果没有岛屿，则返回 0 。
class Solution {
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        int res = 0;

        for(int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    // 如果当前位置是陆地，计算岛屿面积
                    res = max(res, dp(grid, i, j));
                }
            }
        }
        return res;
    }

    // dp 函数定义，输入：当前坐标 (i, j)，返回以 (i, j) 为起点的岛屿面积
    int dp(vector<vector<int>>& grid, int i, int j) {
        // 非法索引检查
        if (i < 0 || j < 0 ||
            i >= grid.size() || j >= grid[0].size()) {
                return 0; // 如果越界，返回面积为 0
        }

        // 如果当前位置是水域，直接返回面积为 0
        if (grid[i][j] == 0) {
            return 0;
        }

        // 将当前位置变成水域，避免重复计算
        grid[i][j] = 0;

        // 计算当前岛屿面积
        return 1 + dp(grid, i-1, j) + dp(grid, i+1, j) +
               dp(grid, i, j-1) + dp(grid, i, j+1);
    }
};

// 5.子岛屿的数量
// 给你一个 m x n 的二进制矩阵 grid1 和 grid2 ，其中 0 代表水域，1 代表陆地。
// 请你返回 grid2 中 子岛屿 的 数目 。子岛屿是指在 grid2 中的一个岛屿，且这个岛屿在 grid1 中也是一个岛屿。
// 换句话说，子岛屿是指在 grid2 中的一个岛屿，且这个岛屿在 grid1 中也是一个岛屿。
// 请你返回 grid2 中 子岛屿 的 数目 。
class Solution {
public:
    int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2) {
        int m = grid1.size();
        int n = grid1[0].size();


        // 遍历grid1和grid2，找到所有的子岛屿
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // 如果grid2中的位置是陆地且grid1中是海水
                if (grid2[i][j] == 1 && grid1[i][j] == 0) {
                    // 进行深度优先搜索，淹没所有相连的陆地
                    dp(grid2, i, j);
                }
            }
        }
        // 现在 grid2 中的陆地已经被淹没，剩下的陆地都是子岛屿
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid2[i][j] == 1) {
                    res++;
                    // 进行深度优先搜索，淹没所有相连的陆地
                    dp(grid2, i, j);
                }
            }
        }

        return res;

    }

    // 从 (i, j) 开始，将与之相邻的陆地都变成海水
    void dp(vector<vector<int>>& grid, int i, int j) {
        // 非法索引检查
        if (i < 0 || j < 0 ||
            i >= grid.size() || j >= grid[0].size()) {
                return;
        }

        // 如果当前位置是水域，直接返回
        if (grid[i][j] == 0) {
            return;
        }

        // 将当前位置变成水域
        grid[i][j] = 0;

        // 淹没上下左右的陆地
        dp(grid, i-1, j); // 上
        dp(grid, i+1, j); // 下
        dp(grid, i, j-1); // 左
        dp(grid, i, j+1); // 右
    }
};

// 6.子集（元素不重复不可复选）
// 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        backtracking(nums, 0);
        return res;
    }

    // 回溯函数
    void backtracking(vector<int>& nums, int start) {
        // 前序位置，每条路径都是子集
        res.push_back(path);

        for (int i = start; i < nums.size(); i++) {
            // 1.做选择
            path.push_back(nums[i]);

            // 2.递归
            // 注意：通过参数start避免子集重复
            backtracking(nums, i+1);

            // 3.撤销选择
            path.pop_back();
        }
    }
};

// 子集Ⅱ（元素可重复不可复选）
// 解题思路：先把原数组排序，让重复元素相邻排列，如果发现 nums[i] == nums[i-1] ，则跳过
class Solution {
private:
     vector<vector<int>> res;
     vector<int> path;
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        // 先对原数组进行排序
        sort(nums.begin(), nums.end());

        backtrack(nums, 0);

        return res;
    }
    
    void backtrack(vector<int>& nums, int start) {
        // 前序位置
        res.push_back(path);

        for (int i = start; i < nums.size(); i++) {
            // 如果当前元素和前一个元素相同，就跳过
            // 注意：这里的第一个判定条件
            if (i > start && nums[i-1] == nums[i]) continue;

            path.push_back(nums[i]);

            backtrack(nums, i+1);

            path.pop_back();
        }
    }
};

// 7.组合
// 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。你可以按 任何顺序 返回答案。
// 解题思路：只保留元素数量符合要求的那一层子集
class Solution {
private:
    vector<int> path;
    vector<vector<int>> res;
public:
    vector<vector<int>> combine(int n, int k) {
        // 注意：从1开始
        backtrack(1, n, k);
        return res;
    }

    void backtrack(int start, int n, int k) {
        // base case
        if (path.size() == k) {
            res.push_back(path);
            // 注意：可以提前终止
            return;
        }

        // 回溯
        // 注意：循环判定条件包含n
        for (int i = start; i <= n; i++) {
            // choose
            path.push_back(i);

            // recurse
            backtrack(i+1, n ,k);

            // 撤销选择
            path.pop_back();
        }
    }
};

// 组合Ⅱ
// 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的每个数字在每个组合中只能使用 一次 。
// 解题思路：
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    int cursum = 0;

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        // 注意：要先排序，才能实现剪支
        sort(candidates.begin(), candidates.end());
        backtrack(candidates, target, 0);
        return res;
    }

    void backtrack(vector<int>& candidates, int target, int start) {
        // base case 
        if (cursum == target) {
            res.push_back(path);
            return;
        }

        // 注意：如果超过目标和超过 target 也直接返回
        if (cursum > target) return;

        for (int i = start; i < candidates.size(); i++) {
            //注意：剪掉数字相同的相邻分支
            if (i > start && candidates[i-1] == candidates[i]) {
                continue;
            }

            path.push_back(candidates[i]);
            cursum += candidates[i];

            backtrack(candidates, target, i+1);

            path.pop_back();
            cursum -= candidates[i];
        }
    }
};


// 回溯算法练习题
// 连续差相同的数字
// 解题思路：看作是元素无重可复选的全排列问题，增加两条剪支逻辑
class Solution {
public:
    // 记录当前路径组成的数字的值
    int track = 0;
    // 记录当前数字的位数
    int digit = 0;

    // 存储结果
    vector<int> res;

    vector<int> numsSameConsecDiff(int n, int k) {
        backtrack(n,k);
        return res;
    }

    void backtrack(int n, int k) {
        // base case
        if (digit == n) {
            res.push_back(track);
            return;
        }

        for (int i = 0; i <=9; i++) {
            // 剪支1:第一位不能是0
            if (digit == 0 && i == 0) continue;
            // 剪支2：相邻的数字差必须是k
            if (digit > 0 && abs(i - track % 10) != k) continue;

            track = track * 10 + i;
            digit++;

            backtrack(n,k);

            track = track / 10;
            digit--;
        }
    }
};

// 电话号码的字母组合
// 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

// 解题思路：首先用一个二维数组存储每个数组能表示的字符
class Solution {
private:
    vector<string> res;
    vector<vector<char>> numToChar = {
        {'a','b','c'},
        {'d','e','f'},
        {'g','h','i'},
        {'j','k','l'},
        {'m','n','o'},
        {'p','q','r','s'},
        {'t','u','v'},
        {'w','x','y','z'}
    };
    string path;
public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return {};
        backTrack(digits, 0);
        return res;
    }

    // 回溯函数
    void backTrack(string digits, int i) {
        // base case
        if (path.size() == digits.size()) {
            res.push_back(path);
            return;
        }

        
        int index = digits[i] - '0' - 2;
        for (auto c : numToChar[index]) {
            // 做选择
            path.push_back(c);
            // 递归
            backTrack(digits, i+1);
            // 撤销选择
            path.pop_back();
        }
    
    }
};