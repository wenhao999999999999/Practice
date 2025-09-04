# include <bits/stdc++.h>
using namespace std;
// 面试经典题：矩阵

// 1.矩阵置零
// 题干：
    // 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
    // 注意：这里时将 0 元素所在的所有元素都置为0，不只是相邻元素

// 思路：
    // 先用两个集合（或布尔数组）记录「需要置零的行」和「需要置零的列」；遍历矩阵，当遇到元素为 0 时，将其所在行和列加入集合；再次遍历矩阵，若元素所在行或列在集合中，则将该元素设为 0。

class Solution {
private:
    unordered_set<int> zero_rows, zero_cols;
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();

        // 第一步：记录需要置零的行和列
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    zero_rows.insert(i);
                    zero_cols.insert(j);
                }
            }
        }

        // 第二步：根据记录置零
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (zero_rows.count(i) || zero_cols.count(j)) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
};

// 2. 生命游戏
// 题干：
    // 根据 百度百科 ， 生命游戏 ，简称为 生命 ，是英国数学家约翰·何顿·康威在 1970 年发明的细胞自动机。
    // 给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态： 1 即为 活细胞 （live），或 0 即为 死细胞 （dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：
    // 如果活细胞周围 八 个位置的活细胞数少于 两个，则该位置活细胞死亡；
    // 如果活细胞周围 八 个位置有 两 个或 三 个活细胞，则该位置活细胞仍然存活；
    // 如果活细胞周围 八 个位置有超过 三 个活细胞，则该位置活细胞死亡；
    // 如果死细胞周围正好有 三 个活细胞，则该位置死细胞复活；

    // 下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是 同时 发生的。给你 m x n 网格面板 board 的当前状态，返回下一个状态。

// 思路：
    // 核心就是“同步更新 + 两位编码”。新状态要同时由旧状态推导：不能一边改一边用改过的结果算邻居。同时存两种状态：
    // board[r][c] 的第 0 位（& 1）存当前状态（0=死，1=活）。第1位（值为2）= 下一轮（新）状态
    // 计算邻居活细胞数 live 时，只看当前位：board[nr][nc] & 1。
    // 根据规则决定下一状态，把它写进第 1 位：若下一步要活，就 board[r][c] |= 2。
    // 第一趟遍历结束后，统一右移一位 >>= 1，把下一状态挪到当前位，完成更新。

// 代码：
class Solution {
public:
    void gameOfLife(vector<vector<int>>& board) {
        // 外壳与边界检查
        int m = board.size();
        if (m == 0) return;
        int n = board[0].size();
        if (n == 0) return;

        // 8 个邻居方向，(dr[k], dc[k]) 依次代表左上、上、右上、左、右、左下、下、右下，便于遍历邻居。
        static const int dr[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
        static const int dc[8] = {-1, 0, 1,-1, 1,-1, 0, 1};

        //第一次双重循环：计算“下一状态”，先写到第1位
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                int live = 0;
                // 对 (r,c) 的 8 个方向做边界判断。
                for (int k = 0; k < 8; ++k) {
                    int nr = r + dr[k], nc = c + dc[k];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                        live += (board[nr][nc] & 1);  // 只看当前状态位
                    }
                }
                int cur = board[r][c] & 1;
                int next = 0;
                // 规则：活细胞 2~3 邻居存活；死细胞邻居恰好 3 复活
                if ((cur == 1 && (live == 2 || live == 3)) ||
                    (cur == 0 && live == 3)) {
                    next = 1;
                }
                if (next) board[r][c] |= 2; // 把下一状态写到第1位
            }
        }

        // 第二次遍历：右移一位，下一状态成为当前状态
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                board[r][c] >>= 1;
            }
        }
    }
};
