// BFS (Breadth-First Search) 算法解题套路框架
#include <bits/stdc++.h>
#include "../include/binaryTree.h"
using namespace std;

// BFS 算法解题框架
// DFS/回溯/BFS 这类算法，本质上就是把具体的问题抽象成树结构，然后遍历这棵树进行暴力穷举，所以这些穷举算法的代码本质上就是树的遍历代码。
// BFS 算法的本质就是遍历一幅图，下面你就会看到了，BFS 的算法框架就是 图结构的 DFS/BFS 遍历 中遍历图节点的算法代码。而图的遍历算法其实就是多叉树的遍历算法加了个 visited 数组防止死循环；多叉树的遍历算法又是从二叉树遍历算法衍生出来的。所以我说 BFS 算法的本质就是二叉树的层序遍历。
// BFS 算法经常用来求解最短路径问题



// 1. 滑动谜题
class Solution {
public:
    // 计算滑动谜题从初始状态board到达目标状态所需的最小步数
    int slidingPuzzle(vector<vector<int>>& board) {
        string target = "123450";

        // 将二维数组board转换为一个字符串表示
        string start = "";
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                 start = start + to_string(board[i][j]);
            }
        }

        // 使用广度优先搜索算法来找到从start到target的最短路径
        queue<string> q;
        unordered_set<string> visited;
        q.push(start);
        visited.insert(start);

        int steps = 0;
        while(!q.empty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                string cur = q.front();
                q.pop();

                // 判断是否到达目标状态
                if (cur == target) {
                    return steps;
                }

                // 将数字0与相邻的数字交换位置，生成所有可能的邻居状态
                for (string neighborBoard : getNeighbors(cur)) {
                    if (!visited.count(neighborBoard)) {
                        q.push(neighborBoard);
                        visited.insert(neighborBoard);
                    }
                }
            }
            steps++;
        }
        return -1; // 如果无法到达目标状态，返回 -1
    }

    // 根据当前状态board生成所有可能的邻居状态
    vector<string> getNeighbors(string board) {
        // 记录一维字符串的相邻索引
        vector<vector<int>> mapping = {
            {1, 3}, // 0
            {0, 2, 4}, // 1
            {1, 5}, // 2
            {0, 4}, // 3
            {1, 3, 5}, // 4
            {2, 4} // 5
        };

        int zeroIndex = board.find('0');
        vector<string> neighbors;
        for (int adj : mapping[zeroIndex]) {
            string neighbor = board;
            swap(neighbor[zeroIndex], neighbor[adj]);
            neighbors.push_back(neighbor);
        }
        return neighbors;
    }
};     


// 2. 打开转盘锁
// 给定一个四位数的转盘锁，初始状态为 "0000"。每次可以将某一位数字加1或减1，0和9是环形的。
// 请你返回打开转盘锁所需的最少步数。每次只能改变一位数字，且只能在 "0000" 到 "9999" 的范围内。
// 列表 deadends 包含了不能到达的状态，target 是目标状态。
// 如果无法到达目标状态，返回 -1。
class Solution {
public:
    int openLock(vector<string>& deadends, string target) {
        // 记录需要跳过的死亡密码
        unordered_set<string> deads(deadends.begin(), deadends.end());

        if (deads.count("0000")) {
            return -1; // 如果初始状态就是死亡密码，直接返回 -1
        }

        // 记录已经穷举过的密码，防止走回头路
        unordered_set<string> visited;
        visited.insert("0000");

        // 使用队列进行广度优先搜索
        queue<string> q;
        q.push("0000");
        
        // 记录当前步数
        int steps = 0;

        while (!q.empty()) {
            int sz = q.size();
            
            // 将当前队列中的所有节点向周围扩散
            for (int i = 0; i < sz; i++) {
                // 取出当前节点
                string cur = q.front();
                q.pop();

                // 判断是否到达终点
                if (cur == target) {
                    return steps;
                }

                // 将一个节点的合法相邻节点加入队列
                for (string neighbor : getNeihbors(cur)) {
                    if (!visited.count(neighbor) &&!deads.count(neighbor)) {
                        q.push(neighbor);
                        visited.insert(neighbor);
                    }

                }
            }

            // 步数加1
            steps++;
        }

        return -1; // 如果无法到达目标状态，返回 -1
    }

    // 根据当前状态board生成所有可能的邻居状态
    vector<string> getNeihbors(string board) {
        vector<string> neighbors;
        for (int i = 0; i < 4; i++) {
            // 尝试将第i位数字加1
            string neighbor = board;
            int digit = board[i] - '0';
            digit = (digit + 1) % 10;
            neighbor[i] = digit + '0';
            neighbors.push_back(neighbor);

            // 尝试将第 i 位数字减1
            neighbor = board;
            digit = board[i] - '0';
            digit = (digit + 9) % 10;
            neighbor[i] = digit + '0';
            neighbors.push_back(neighbor);
        }
        return neighbors;
    }
};

// 3.完全二叉树插入器
// 完全二叉树 是每一层（除最后一层外）都是完全填充（即，节点数达到最大）的，并且所有的节点都尽可能地集中在左侧。
// 设计一种算法，将一个新节点插入到一棵完全二叉树中，并在插入后保持其完整。
// 实现 CBTInserter 类:
// CBTInserter(TreeNode root) 使用头节点为 root 的给定树初始化该数据结构；
// CBTInserter.insert(int v)  向树中插入一个值为 Node.val == val的新节点 TreeNode。使树保持完全二叉树的状态，并返回插入节点 TreeNode 的父节点的值；
// CBTInserter.get_root()  返回树的头节点。

class CBTInserter {
private:
    TreeNode* root;
    queue<TreeNode*> q;
public:
    // 构造函数，初始化根节点并使用广度优先搜索找到所有可以插入子节点的位置
    CBTInserter(TreeNode* root) {
        this->root = root;

        // BFS 找到底部可插入的节点
        queue<TreeNode*> temp;
        temp.push(root);
        while (!temp.empty()) {
            TreeNode* node = temp.front();
            temp.pop();
            if (node->left) {
                temp.push(node->left);
            }
            if (node->right) {
                temp.push(node->right);
            }

            if (node->left == nullptr || node->right == nullptr) {
                // 找到完全二叉树的底部底部可以插入的节点
                q.push(node);
            }
        
        }
    }
    
    // 插入新节点到完全二叉树中，并返回插入节点的父节点值
    int insert(int v) {
        TreeNode* node = new TreeNode(v);
        TreeNode* cur = q.front();

        // 进行插入
        if (cur->left == nullptr) {
            cur->left = node;
        } else if (cur->right == nullptr) {
            cur->right = node;
            // 当且仅当插入节点是父节点的右孩子时，才需要更新父节点的值
            q.pop();
        }

        // 新节点的左右节点也是可以插入的
        q.push(node);
        return cur->val;

    }

    // 返回完全二叉树的根节点
    TreeNode* get_root() {
        return root;
    }
};


// 4.填充每个节点的下一个右侧节点指针
// 给定一个二叉树：
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
};
// 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL 。
// 初始状态下，所有 next 指针都被设置为 NULL 。

class Solution {
public:
    Node* connect(Node* root) {
        if (!root) {
            return nullptr;
        }

        // 初始化队列
        queue<Node*> q;

        // 根节点入队
        q.push(root);

        // 每一层第一个节点的 next 指针指向 NULL
        Node* prev = nullptr;
        // BFS
        while (!q.empty()) {
            int sz = q.size();

            for (int i = 0; i < sz; i++) {
                Node* cur = q.front();
                q.pop();
                // 如果前序节点不是每一层第一个节点，则将 prev 节点的 next 指针指向当前节点
                if (prev) {
                    prev->next = cur;
                }

                prev = cur;

                // 左右孩子入队
                if (cur->left) {
                    q.push(cur->left);
                }

                if (cur->right) {
                    q.push(cur->right);
                }
            }

            prev = nullptr;
        }
        return root;
    }
};

// 5. 二叉树的最大宽度
// 每一层的 宽度 被定义为该层最左和最右的非空节点（即，两个端点）之间的长度。将这个二叉树视作与满二叉树结构相同，两端点间会出现一些延伸到这一层的 null 节点，这些 null 节点也计入长度。
// 给你一个二叉树的根节点 root ，返回其最大宽度。

// 完全二叉树编号+层序遍历解法：
class Solution {
public: 
    // 记录节点和对应的编号
    struct Pair {
        TreeNode* node;
        unsigned long long id;

        Pair(TreeNode* node, unsigned long long id): node(node), id(id) {}
    };
    int widthOfBinaryTree(TreeNode* root) {
        if (!root) return 0;

        // 记录最大宽度
        int maxWidth = 0;

        // 层序遍历
        queue<Pair> q;
        q.push(Pair(root, 1));
        while (!q.empty()) {
            int sz = q.size();
            
            // 注意：起始编号和结束编号的数据类型要用 unsigned long long，否则会溢出
            unsigned long long start = 0;
            unsigned long long end = 0;

            for (int i = 0; i < sz; i++) {
                Pair cur = q.front();
                q.pop();

                // 队首元素是当前层的第一个元素
                if (i == 0) start = cur.id;

                // 队尾元素是当前层的最后一个元素
                if (i == sz - 1) end = cur.id;

                if (cur.node->left) {
                    q.push(Pair(cur.node->left, 2 * cur.id));
                }

                if (cur.node->right) {
                    q.push(Pair(cur.node->right, 2 * cur.id + 1));
                }
            }

            // 计算当前层的宽度
            unsigned long long width = end - start + 1;

            // 更新最大宽度
            maxWidth = max(maxWidth, static_cast<int>(width));
        }

        return maxWidth;
    }
};

// 6.最小基因变化
// 题目描述：基因序列可以表示为一条由 8 个字符组成的字符串，其中每个字符都是 'A'、'C'、'G' 和 'T' 之一。
// 假设我们需要调查从基因序列 start 变为 end 所发生的基因变化。一次基因变化就意味着这个基因序列中的一个字符发生了变化。例如，"AACCGGTT" --> "AACCGGTA" 就是一次基因变化。
// 另有一个基因库 bank 记录了所有有效的基因变化，只有基因库中的基因才是有效的基因序列。（变化后的基因必须位于基因库 bank 中）
// 给你两个基因序列 start 和 end ，以及一个基因库 bank ，请你找出并返回能够使 start 变化为 end 所需的最少变化次数。如果无法完成此基因变化，返回 -1 。
// 注意：起始基因序列 start 默认是有效的，但是它并不一定会出现在基因库中。
// 解题思路：
class Solution {
public:
    int minMutation(string startGene, string endGene, vector<string>& bank) {
        // 将基因库存入哈希表便于快速查找
        unordered_set<string> bankSet(bank.begin(), bank.end());
        if (bankSet.find(endGene) == bankSet.end()) {
            return -1;
        }
        vector<char> AGCT = {'A', 'G', 'C', 'T'};

        // BFS 标准框架
        queue<string> q;
        unordered_set<string> visited;
        q.push(startGene);

        // 注意：这里是用哈希表去标记可能的基因序列，不是数组
        visited.insert(startGene);
        int step = 0;
        while (!q.empty()) {
            // 注意：计算队列中元素的个数应该在循环内，因为队列中元素的个数会动态变化
            int sz = q.size();
            for (int j = 0; j < sz; j++) {
                string cur = q.front();
                q.pop();
                if (cur == endGene) {
                    return step;
                }
                // 向周围扩散
                for (const string& newGene : getAllMutation(cur)) {
                    if (visited.find(newGene) == visited.end() && bankSet.find(newGene) != bankSet.end()) {
                        q.push(newGene);
                        visited.insert(newGene);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    // 当前基因的每个位置都可以变异为 A/G/C/T，穷举所有可能的结构
    vector<string> getAllMutation(const string& gene) {
        vector<string> res;
        string geneChars = gene;
        for (int i = 0; i < geneChars.length(); i++) {
            char oldChar = geneChars[i];
            for (char newChar : vector<char>{'A', 'G', 'C', 'T'}) {
                geneChars[i] = newChar;
                res.push_back(geneChars);
            }
            // 注意：变换为某个位置的基因后要恢复成原来的基因
            geneChars[i] = oldChar;
        }
        return res;
    }
};