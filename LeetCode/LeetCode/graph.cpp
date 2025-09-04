#include <bits/stdc++.h>
using namespace std;

// 图经典习题

// 1.被围绕的区域
// 题干：
    // 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' 组成，捕获 所有 被围绕的区域：
    // 连接：一个单元格与水平或垂直方向上相邻的单元格连接。
    // 区域：连接所有 'O' 的单元格来形成一个区域。
    // 围绕：如果您可以用 'X' 单元格 连接这个区域，并且区域中没有任何单元格位于 board 边缘，则该区域被 'X' 单元格围绕。
    // 通过 原地 将输入矩阵中的所有 'O' 替换为 'X' 来 捕获被围绕的区域。你不需要返回任何值。
// DFS 思路：
    // 关键点在于识别哪些 'O' 区域是未被围绕的（与边界相连），哪些是被围绕的（内部区域）。
    // 直接寻找被围绕的区域比较困难，因此采用逆向思维：先找到所有与边界相连的 'O' 区域（这些区域不会被围绕），并标记它们。
class Solution {
public:
    void solve(vector<vector<char>>& board) {
        if (board.empty()) return;
        int m = board.size();
        int n = board[0].size();

        // 步骤一：将边界 'O' 元素改成 'A'
        for (int i = 0; i < m; i++) {
            dfs(board, i, 0); // 左边界
            dfs(board, i, n-1); // 右边界
        }

        for (int j = 0; j < n; j++) {
            dfs(board, 0, j); // 上边界
            dfs(board, m-1, j); // 下边界
        }


        // 步骤二：将所有'O'字符改成'X'
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                }
            }
        }

        
    }

    // dfs 淹没所有与i，j相邻的元素
    void dfs(vector<vector<char>>& board, int i, int j) {
        int m = board.size();
        int n = board[0].size();

        // 检查索引是否超边界
        if (i < 0 || j < 0 || i >= m || j >= n)
            return;

        // 若当前元素不是'O'，直接返回
        if (board[i][j] == 'X')
            return;

        board[i][j] = 'A';

        // 递归淹没连接区域
        dfs(board, i - 1, j);
        dfs(board, i + 1, j);
        dfs(board, i, j - 1);
        dfs(board, i, j + 1);
    }
};

// BFS思路：
    // 步骤一：将边界 'O' 元素加入队列并标记为 'A'；
    // 步骤二：BFS处理所有与边界相连的 'O'，标记为 'A'；
    // 步骤三：替换字符（'O'→'X'，'A'→'O'）
class Solution {
public:
    void solve(vector<vector<char>>& board) {
        // 快速判断
        if (board.empty()) return;

        int m = board.size();
        int n = board[0].size();

        // 
        queue<pair<int, int>> q;

        // 步骤一：将边界 'O' 元素加入队列并标记为 'A'
        // 处理左右边界
        for (int i = 0; i < m; ++i) {
            if (board[i][0] == 'O') {
                board[i][0] = 'A';
                q.push({i, 0});
            }
            if (board[i][n-1] == 'O') {
                board[i][n-1] = 'A';
                q.push({i, n-1});
            }
        }
        // 注意点1：处理上下边界时跳过角落，因为角落已经在左右边界处理过，避免重复。
        for (int j = 1; j < n-1; ++j) {
            if (board[0][j] == 'O') {
                board[0][j] = 'A';
                q.push({0, j});
            }
            if (board[m-1][j] == 'O') {
                board[m-1][j] = 'A';
                q.push({m-1, j});
            }
        }

        // 方向数组：上下左右
        vector<pair<int, int>> dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};

        // 步骤二：BFS处理所有与边界相连的 'O'，标记为 'A'
        while (!q.empty()) {
            auto [i, j] = q.front();
            q.pop();
            // 遍历四个方向
            for (auto [di, dj] : dirs) {
                int x = i + di;
                int y = j + dj;
                // 检查是否在矩阵范围内，且是未标记的 'O'
                if (x >= 0 && x < m && y >= 0 && y < n && board[x][y] == 'O') {
                    board[x][y] = 'A';  // 标记为 'A'
                    q.push({x, y});    // 加入队列继续处理
                }
            }
        }

        // 步骤三：替换字符（'O'→'X'，'A'→'O'）
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                }
            }
        }
    }
};

// 2.所有可能的路径
// 题干：
    // 给你一个有 n 个节点的 有向无环图（DAG），请你找出从节点 0 到节点 n-1 的所有路径并输出（不要求按特定顺序）
    //  graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph[i][j]存在一条有向边）。
// 思路： 
    // 由于图是无环的，不需要担心循环或重复访问的问题，因此可以使用深度优先搜索（DFS）来遍历所有可能路径。
    // 
class Solution {
private:
    // 使用一个动态数组 path 记录当前路径上的节点，当到达目标节点时，将当前路径保存到结果数组 res 中。 
    vector<int> path; 
    vector<vector<int>> res;
public:
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        int n = graph.size();
        traverse(graph, 0, n - 1);
        return res;
    }
    // dfs
    void traverse(vector<vector<int>>& graph, int src, int end) {
        
        // 前序位置
        path.push_back(src);
        
        // base case
        if (src == end) {
            res.push_back(path);
            // 注意：找到一条路径后需要将当前结点弹出，因为可能存在其他路径到达中终点
            path.pop_back();
            return;
        }

        // 层序遍历邻居节点
        for (auto node : graph[src]) {
            traverse(graph, node, end);
        }

        // 后序位置
        // 注意：到达当前结点可能存在其他路径，因此需要弹出当前结点。
        path.pop_back();

    }
};

// 3. 克隆图
// 题干：
    // 给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

    // Definition for a Node.
    class Node {
    public:
        int val;
        vector<Node*> neighbors;
        Node() {
            val = 0;
            neighbors = vector<Node*>();
        }
        Node(int _val) {
            val = _val;
            neighbors = vector<Node*>();
        }
        Node(int _val, vector<Node*> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    };

    // 简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（val = 1），第二个节点值为 2（val = 2），以此类推。该图在测试用例中使用邻接列表表示。
    // 邻接列表 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。 
    // 给定节点将始终是图中的第一个节点（值为 1）。你必须将 给定节点的拷贝 作为对克隆图的引用返回。
// 思路：
    // 深拷贝意味着每个节点都是新创建的，而不是原始节点的引用，并且新图的结构与原始图完全相同。使用深度优先搜索（DFS）来遍历原始图，并同时构建克隆图。
    // 先创建新结点，再构建连接关系
// 代码：
class Solution {
public:
    Node* cloneGraph(Node* node) {
        // DFS 遍历图，顺便构建克隆图
        traverse(node);
        // 从 map 里找到克隆图的对应节点
        return originToClone[node];
    }

private:
    // 记录 DFS 遍历过的节点，防止走回头路
    unordered_set<Node*> visited;
    // 记录原节点到克隆节点的映射
    unordered_map<Node*, Node*> originToClone;

    // DFS 图遍历框架
    void traverse(Node* node) {
        // base case

            // 如果节点为空，直接返回；
            if (node == nullptr) {
                return;
            }
            // 如果节点已经被访问过，也返回（避免重复处理）
            if (visited.count(node)) {
                return;
            }

        // 前序位置，标记为已访问
        visited.insert(node);

        // 前序位置，克隆节点
        if (!originToClone.count(node)) {
            originToClone[node] = new Node(node->val);
        }
        Node* cloneNode = originToClone[node];

        // 递归遍历邻居节点，并构建克隆图
        for (Node* neighbor : node->neighbors) {
            traverse(neighbor);
            // 递归结束后，邻居节点的克隆一定存在于映射中（因为递归会先处理邻居节点
            Node* cloneNeighbor = originToClone[neighbor];
            // 构建邻居关系：从映射中获取邻居节点的克隆，并将其添加到当前克隆节点的邻居列表中。
            cloneNode->neighbors.push_back(cloneNeighbor);
        }
    }
};