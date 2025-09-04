// 二叉树的递归分为「遍历」和「分解问题」两种思维模式
// 分解思维模式：把规模较大的问题 fib(5) 分解成规模较小的问题 fib(4) 和 fib(3)，然后通过子问题的解得到原问题的解，我们可以称这种思维模式为「分解问题」。如果你想用「分解问题」的思维模式来写递归算法，那么这个递归函数一定要有一个清晰的定义，说明这个函数参数的含义是什么，返回什么结果。
// 遍历思维模式：递归树上的节点并没有一个明确的含义，只是记录了之前所做的一些选择。所有全排列，就是所有叶子节点上的结果。这种思维模式称为「遍历」。如果你想用「遍历」的思维模式来写递归算法，那么你需要一个无返回值的遍历函数，在遍历的过程中收集结果。
// 二叉树经典习题
# include<bits/stdc++.h>
using namespace std;

 struct TreeNode {
     int val;
     TreeNode *left;    
     TreeNode *right;
     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

// 1.二叉树的所有路径
// 题目描述：
    // 给你一个二叉树的根节点 root ，按 任意顺序 ，返回所有从根节点到叶子节点的路径。叶子节点 是指没有子节点的节点。
// 解题思路：
    // 采用 DFS 算法，用一个存储字符串元素的动态数组作为返回值，用一个存储字符串元素的动态数组存储二叉树每条路径上的字符，当达到叶子节点是把路径压入结果；
class Solution {
private:
    // 记录当前二叉树路径
    vector<string> path;

    // 记录所有二叉树路径
    vector<string> res;
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        if (!root) return{};
        traverse(root);
        return res;
    }

    // 递归遍历函数
    void traverse (TreeNode* root) {
        // base case
        if (!root) return;

        // 判断当前路径是否走到叶子结点
        if (root->left == nullptr && root->right == nullptr) {
            // 先压入当前结点
            path.push_back(to_string(root->val));
            // 再把当前路径压入结果中
            res.push_back(joinPath(path));

            // 注意点4：这里要撤销选择
            path.pop_back();

            // 注意点5：找到一条路径之后要返回
            return;
        }

        // 前序位置，将字符压入当前路径
        // 注意点1: 数值类型(整数类型：int、long、long long、unsigned int、unsigned long、unsigned long long浮点类型：float、double、long double) 转 sting 函数 to_string()
        path.push_back(to_string(root->val));

        // 递归遍历左子树
        traverse(root->left);
        // 递归遍历右子树
        traverse(root->right);

        // 注意点3：后序位置，撤销选择
        path.pop_back();
    }

    // 规范格式函数
    string joinPath (const vector<string> path) {
        // 字符串流
        stringstream ss;

        // 遍历字符串，合并路径
        for (int i = 0; i < path.size(); i++) {
            ss << path[i];
            // 注意点2：字符串流用于字符串和其他数据类型之间的转换：将其他类型（如 int、double 等）转换为字符串；将字符串转换为数值；拼接不同类型的数据为字符串
            if (i < path.size() -1 ) ss << "->";
        }

        return ss.str(); // 通过 str() 方法获取转换后的字符串
    } 
};

// 2.求根节点到叶节点数字之和
class Solution {
private:
    // 存放路径元素
    vector<int> path;

    // 存放每条路径的和
    vector<int> sum;

    // 返回结果
    int res;
public:
    int sumNumbers(TreeNode* root) {
        if (!root) return 0;
        traverse(root);
        for (auto n : sum) {
            res += n;
        }
        return res;
        
    }

    // 递归遍历，结合回溯算法
    void traverse(TreeNode* root) {
        // 注意：这里要结点为空直接返回
        if (!root) return;
        // base case
        if (root->left == nullptr && root->right == nullptr) {
            path.push_back(root->val);
            sum.push_back(calculateSum(path));
            path.pop_back();
            return;
        }

        // 前序位置
        path.push_back(root->val);

        // 递归遍历左右子树
        traverse(root->left);
        traverse(root->right);

        // 后序位置
        path.pop_back();

    }

    // 计算每条路径上结点的数并返回
    int calculateSum(vector<int> path) {
        int nums = 0;
        for (auto num : path) {
            nums = nums*10+num;
        }

        return nums;
    }
};

// 3.二叉树的右视图
//  给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
// BFS解法：结果数组中保存每一层结点的第一个结点的值（从右往左遍历）
class Solution {
private:
    // 存储结果数组
    vector<int> res = {};
    
    // 队列：存储每一层结点
    queue<TreeNode*> q;
public:
    vector<int> rightSideView(TreeNode* root) {
        if (root == nullptr) return res;

        q.push(root);

        while (!q.empty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                TreeNode* curNode = q.front();
                q.pop();
                // 只保存每一层第一个结点的值
                if (i == 0) res.push_back(curNode->val);

                // 将当前节点的左右子树加入队列
                // 先压入右子树，后压入左子树
                if (curNode->right) q.push(curNode->right);
                if (curNode->left) q.push(curNode->left);
            }
        }

        return res;
    }
};

// 4.从叶节点开始的最小字符串
// 给定一颗根结点为 root 的二叉树，树中的每一个结点都有一个 [0, 25] 范围内的值，分别代表字母 'a' 到 'z'。返回 按字典序最小 的字符串，该字符串从这棵树的一个叶结点开始，到根结点结束。

// 递归遍历解法
class Solution {
private:
    // 存储当前路径
    string path;
    // 存储最短字符串
    string res;    
public:
    string smallestFromLeaf(TreeNode* root) {
        traverse(root);
        return res;
    }

    // 递归遍历函数
    void traverse (TreeNode* root) {
        if (!root) return;

        // base case
        if (root->left == nullptr && root->right == nullptr) {
            // 前序位置，压入元素
            path.push_back('a' + root->val);

            // 反转字符串
            reverse(path.begin(), path.end());

            // 第一次遍历或者当前路径字符串更小
            if (res.empty() || res > path) {
                res = path;
            }

            // 注意：恢复，正确维护 path 中的元素
            reverse(path.begin(), path.end());

            // 注意：后序位置，撤销选择
            path.pop_back();
            return;
        }

        // 前序位置
        path.push_back('a' + root->val);

        // 递归遍历左右子树
        traverse(root->left);
        traverse(root->right);

        // 注意：后序位置，撤销选择
        path.pop_back();
    }
};

// 5.从根到叶的二进制数之和
// 给出一棵二叉树，其上每个结点的值都是 0 或 1 。每一条从根到叶的路径都代表一个从最高有效位开始的二进制数。对树上的每一片叶子，我们都要找出从根到该叶子的路径所表示的数字。

// DFS解法：首先求出每一条路径，接着将每一条路径转换为十进制数，最后求和
class Solution {
private:
    // 存储当前路径的结点
    vector<TreeNode*> path;

    // 存储结果
    int sum = 0;

    // 存储所有路径的十进制数
    vector<int> nums;
public:
    // 主函数入口 
    int sumRootToLeaf(TreeNode* root) {
        traverse(root);
        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
        }
        return sum;
    }

    // DFS
    void traverse(TreeNode* root) {
        // 判断当前节点是否为空
        if (root == nullptr) return;

        // base case
        if (root->left == nullptr && root->right == nullptr) {
            // 前序位置，压入结点
            path.push_back(root);

            // 将当前路径转换为十进制数字并保存
            nums.push_back(pathToNum(path));

            // 后序位置，弹出当前结点
            path.pop_back();
            return;
        }

        // 前序位置，压入结点
        path.push_back(root);

        // 递归遍历左右子树
        traverse(root->left);
        traverse(root->right);

        // 后序位置，弹出结点
        path.pop_back();
    }
    
    // 将每一条结点路径转换为十进制数字
    int pathToNum (vector<TreeNode*> path) {
        int num = 0;
        for (int i = 0; i < path.size(); i++) {
            // 注意：第一个结点是高位
            num += path[i]->val *  pow(2, path.size()-i-1) ; 
        }
        return num;
    }
};

// 6. 相同的树
// 题目描述：
    // 给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
// 解题思路：
    // 判断两棵树是否相同，可以分解为判断根节点是否相同，然后判断左右子树的结点是否相同。
class Solution {
public:
    // 函数定义：输入两个根节点，返回以它们为根的两棵二叉树是否相同
    bool isSameTree(TreeNode* p, TreeNode* q) {
        // 1. 首先判断两棵树的根节点是否相同

            // 如果两棵树的根节点都为空则相同
            if (p == nullptr && q == nullptr) return true;

            // 如果仅一棵树的根节点为空则不同
            if (p == nullptr || q == nullptr) return false;

            // 如果根节点都不为空但值不相等则不同
            if (p->val != q->val) return false;

        // 2. 然后判断两棵树的左右子树是否相同
            return isSameTree(p->left,q->left) && isSameTree(p->right, q->right);
    }
};

// 7. 对称二叉树
// 题目描述：
    // 给你一个二叉树的根节点 root ， 检查它是否轴对称。
// 解题思路：
    // 首先判断根节点是否为空，然后递归判断左右子树是否轴对称
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if (!root) return true;

        return check(root->left, root->right);
    }

    // 定义：判断输入的两棵树是否是镜像对称的
    bool check (TreeNode* left, TreeNode* right) {
        // 判断左右子树是空的情况
        if (left == nullptr || right == nullptr) return left == right;

        // 两个根结点需要相同
        if (left->val != right->val) return false;

        // 左右子树也要镜像对称
        return check(left->left, right->right) && check(left->right, right->left);
    }
};


// 8. 路经总和
// 题目描述：
    // 给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。叶子节点 是指没有子节点的节点。
// 解题思路：DFS
class Solution {
private:
    vector<int> nums;
    bool flag = false;
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (!root) return false;
        traverse(root, targetSum);
        return flag;
    }

    void traverse (TreeNode* root, int targetSum) {
        // base case
        if (root == nullptr) return;

        if (root->left == nullptr && root->right == nullptr) {
            nums.push_back(root->val);
            int sum = calculateSum(nums);
            if (sum == targetSum) flag = true;
            nums.pop_back();
            return;
        }

        // 前序位置
        nums.push_back(root->val);

        // 递归
        traverse(root->left, targetSum);
        traverse(root->right, targetSum);

        // 后序位置
        nums.pop_back();
    }

    int calculateSum (const vector<int>& nums) {
        int sum = 0;
        for (auto num : nums) {
            sum += num;
        }
        return sum;
    }
};

// 二叉树的最大直径
class Solution {
    int maxDiameter = 0;

public:
    int diameterOfBinaryTree(TreeNode* root) {
        maxDepth(root);
        return maxDiameter;
    }

private:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        int leftMax = maxDepth(root->left);
        int rightMax = maxDepth(root->right);
        // 后序遍历位置顺便计算最大直径
        maxDiameter = std::max(maxDiameter, leftMax + rightMax);
        return 1 + std::max(leftMax, rightMax);
    }
};

// 9. 二叉树中的最大路径和
// 题目描述：
    // 二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。路径和 是路径中各节点值的总和。给你一个二叉树的根节点 root ，返回其 最大路径和 。
// 解题思路： 
    // 分治思想，最大路径和 = 左子树最大单边路径和 + 右子树最大单边路径和 + 根节点值。定义一个最大单边路径和
class Solution {
private:
    // 注意点3：最大路径和需要初始化为INT_MIN而不是0 
    int maxpathsum = INT_MIN;
public:
    int maxPathSum(TreeNode* root) {
        oneSideMax(root);
        return maxpathsum;
    }
    // 定义：输入一棵树的节点，返回以当前结点为起点的最大单边路径和
    int oneSideMax (TreeNode* root) {
        // base case
        if (root == nullptr) return 0;

        // 注意点2：如何当前路径和是负数需要忽略
        int leftmax = max(0, oneSideMax(root->left));
        int rightmax = max(0, oneSideMax(root->right));

        int cur = leftmax + rightmax + root->val;

        maxpathsum = max (maxpathsum, cur);

        // 注意点1：最大单边路径和需要加当前结点的值
        return max(leftmax, rightmax) + root->val;
    }
};

// 10. 二叉搜索树迭代器
// 题目描述：
    // 实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：
    // BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 的根节点 root 会作为构造函数的一部分给出。指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。
    // boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则返回 false 。
    // int next()将指针向右移动，然后返回指针处的数字
    // 注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返回 BST 中的最小元素。你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序遍历中至少存在一个下一个数字。
// 解题思路：
    // 利用栈模拟递归对 BST 进行中序遍历
// 核心代码：
class BSTIterator {
    // 模拟递归栈
    private:
        // 使用栈来模拟递归调用栈，存储待访问的节点
        stack<TreeNode*> stk;

        // 左侧树枝一撸到底
        void pushLeftBranch(TreeNode* p) {
            while (p != nullptr) {
                // 将沿途所有节点压入栈中，这确保了栈顶元素总是当前最小的未访问节点
                stk.push(p);
                p = p->left;
            }
        }

    public:
        BSTIterator(TreeNode* root) {
            // 初始化迭代器，从根节点开始，将所有左子节点压入栈
            pushLeftBranch(root);
        }

        int next() {
            TreeNode* p = stk.top();
            stk.pop();
            pushLeftBranch(p->right);
            return p->val;
        }

        bool hasNext() {
            return !stk.empty();
        }
};

// 11.完全二叉树的节点个数
// 题目描述
    // 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层（从第 0 层开始），则该层包含 1~ 2h 个节点。
// 解题思路：
    // 一棵完全二叉树的两棵子树，至少有一棵是满二叉树：计算满二叉树的节点个数不用一个个节点去数，可以直接通过树高算出来，这也是这道题提高效率的关键点。
// 核心代码：
#include <cmath>

class Solution {
public:
    int countNodes(TreeNode* root) {
        TreeNode* l = root;
        TreeNode* r = root;
        // 记录左、右子树的高度
        int hl = 0, hr = 0;
        while (l != nullptr) {
            l = l->left;
            hl++;
        }
        while (r != nullptr) {
            r = r->right;
            hr++;
        }
        // 如果左右子树的高度相同，则是一棵满二叉树
        if (hl == hr) {
            return (int)pow(2, hl) - 1;
        }
        // 如果左右高度不同，则按照普通二叉树的逻辑计算
        return 1 + countNodes(root->left) + countNodes(root->right);
    }
};

// 12. 二叉树的最近公共祖先
// 题目描述：
    // 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
// 解题思路：
    // 递归函数的定义：给该函数输入三个参数 root，p，q，它会返回一个节点：
    // 情况 1，如果 p 和 q 都在以 root 为根的树中，函数返回的即是 p 和 q 的最近公共祖先节点。
    // 情况 2，那如果 p 和 q 都不在以 root 为根的树中怎么办呢？函数理所当然地返回 null 呗。
    // 情况 3，那如果 p 和 q 只有一个存在于 root 为根的树中呢？函数就会返回那个节点。
// 核心代码：
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // base case
        if (root == nullptr) return nullptr;
        if (root == p || root == q) return root;

        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);
        // 情况 1：如果 p 和 q 都在以 root 为根的树中，那么 left 和 right 一定分别是 p 和 q（从 base case 看出来的）。
        if (left != nullptr && right != nullptr) {
            return root;
        }
        // 情况 2：如果 p 和 q 都不在以 root 为根的树中，直接返回 null。
        if (left == nullptr && right == nullptr) {
            return nullptr;
        }
        // 情况 3：如果 p 和 q 只有一个存在于 root 为根的树中，函数返回该节点。
        return left == nullptr ? right : left;
    }
};

// 13. 二叉树锯齿形层序遍历
// 题目描述：
    // 给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
// 思路：
    // 在常规二叉树层序遍历基础上，设置一个布尔变量 flag ，用于控制遍历方向,用一个双端队列容器存储每一层结点的值。
// 核心代码：
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        // 注意点1：首先判断根节点是否为空，若根节点为空，直接返回空数组；
        if (root == nullptr) return res;
        queue<TreeNode*> q;
        bool flag = true;

        q.push(root);
        while (!q.empty()) {
            int sz = q.size();
            // 注意点2：存储每一层结点的容器要在大循环中清空
            // 注意点3： 用双端队列存储每一层元素
            deque<int> path;
            for (int i = 0; i < sz; i++) {
                TreeNode* cur = q.front();
                q.pop();
                if (flag) {
                    path.push_back(cur->val);
                } else {
                    path.push_front(cur->val);
                }
                
                
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
             
            }
            // 注意点4：强制类型转换
            res.push_back(vector<int>(path.begin(), path.end()));
            flag = !flag;
        }
        return res;
    }
};

// 14. 二叉搜索树的最小绝对差
// 题干：
    // 给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。差值是一个正数，其数值等于两值之差的绝对值。
// 思路：
    // 把 BST 看作一个有序数组，最小绝对差一定出现在相邻的两个元素之间。
// 核心代码：
class Solution {
public:
    TreeNode* pre;
    int res = INT_MAX;
    int getMinimumDifference(TreeNode* root) {
        if (!root) return 0;
        traverse(root);
        return res;
    }

    void traverse (TreeNode* root) {
        // base case
        if (root == nullptr) return;

        traverse(root->left);

        // 中序位置
        // 注意点1：更新最小值前需要判断 pre 指针是否为空
        if (pre != nullptr) {
            // 注意点3：ST的中序遍历会得到一个升序的序列，当前节点值 root->val 总是大于前一个节点值 pre->val
            res = min (res, root->val - pre->val);
        }
        // 注意点2：更新前驱指针 pre 要在中序遍历位置
        pre = root;
        traverse(root->right);
    }
};