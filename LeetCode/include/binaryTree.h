//声明二叉树类的变量和方法
#ifndef BINARY_TREE_H
#define BINARY_TREE_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <string>
using namespace std;

class BinaryTree {
private:
    struct Node {
        int val;
        Node* left;
        Node* right;
        Node* next;
        Node(int x);
    };

    //二叉树根节点
    Node* root;

    //二叉树遍历到的节点的深度
    int depth = 0;

    //二叉树的最大深度
    int maxdepth = 0; 

    //二叉树的最大直径
    int maxdiameter = 0;

    //二叉树的前序遍历结果，递归思路
    vector<int> preordertraversal;

    //返回二叉树前序遍历结果，分治思路
    vector<int> preOrderTraversal(Node* root);

    //递归遍历二叉树
    void traverse(Node* root);

    //统计二叉树节点数目
    int countInside(Node* root);

    //计算二叉树最大深度
    int maxDepth1(Node* root);
    int maxDepth2(Node* root);

    //计算二叉树直径 
    int diameterOfBinaryTree(Node* root);

    //二叉树展开为链表
    void flatten(Node* root);

    //填充每一个结点的下一个右侧结点的指针
    Node* connect(Node* root);

    // 返回二叉树的序列化字符串
    string serialize(Node* root);

    // 寻找重复的子树
    vector<Node*> findDuplicateSubtrees(Node* root);

    // 记录所有子树的序列化结果,以及出现次数
    unordered_map<string, int> subTrees;

    // 记录重复的子树根节点
    vector<Node*> duplicateSubtreesRoot;

public:
    //二叉树对象构造函数
    BinaryTree();

    //输入一个数组，创建一颗二叉树
    void buildBinaryTree1(vector<int>& values);

    //构造最大二叉树辅助函数
    Node* build1 (vector<int>& nums, int lo, int hi);

    //输入一个数组，按照最大二叉树原则，递归地构建一棵二叉树
    Node* constructMaximumBinaryTree(vector<int>& nums);

    // 存储 inorder 中值到索引的映射
    unordered_map<int, int> valToIndex;

    //根据前序遍历和中序遍历构造二叉树辅助函数
    Node* build2(vector<int> preorder, int preStart, int preEnd,
                 vector<int> inorder, int inStart, int inEnd);

    Node* buildBinaryTree2(vector<int> preorder, vector<int>& inorder);

    //根据中序遍历和后序遍历结果构造二叉树，辅助函数
    Node* build3(vector<int>& inorder, int inStart, int inEnd,
                 vector<int>& postorder, int postStart, int postEnd);

    Node* buildBinaryTree3(vector<int>& inorder, vector<int>& postorder);

    //根据前序遍历和后序遍历构造二叉树，辅助函数
    Node* build4(vector<int> preorder, int preStart, int preEnd,
                 vector<int>& postorder, int postStart, int postEnd);
    
    //根据前序遍历和后序遍历构造二叉树，存在多个答案可以返回任何一个
    Node* buildBinaryTree4(vector<int>& preorder, vector<int>& postorder);

    //二叉树的层序遍历
    vector<vector<int>> levelOrder();

    //二叉树的递归遍历
    void traverse();

    //返回二叉树结点数目
    int count();

    //返回二叉树的最大深度
    int maxDepth();

    //返回二叉树前序遍历结果
    vector<int> PreOrderTraversal();

    //返回二叉树的直径
    int diameterOfBinaryTree();

    //三叉树遍历框架
    void TernaryTreeTraversal (Node* node1, Node* node2);
};

// 力扣的二叉树节点的定义
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}; 
};

// 二叉树序列化的类
class Codec {
public: 
    // 定义分隔符,用于分开每个节点
    string SEP = ",";

    //定义空指针，用于标记 NULL
    string NULLSYM = "#";

    // 把一颗二叉树序列化成字符串,遍历方法实现
    string serializeTraverse(TreeNode* root);
    void _serializeTraverse(TreeNode* root, string& sb);

    // 把字符串反序列化成二叉树,遍历方法实现
    TreeNode* deserializeTraverse(string data);
    TreeNode* _deserializeTraverse(list<string>& nodes);
};

// 二叉搜索树的类
class BinarySearchTree {
public:
    // 当前节点的排名
    int rank = 0;

    // 第 K 小的元素
    int res = 0;

    // 累加和
    int sum = 0;

    // 1.二叉搜索树中第 K 小的元素
        // 给定一个二叉搜索树的根节点 root ，和一个整数 k ，
        // 请你设计一个算法查找其中第 k 小的元素（从 1 开始计数）。
    int kthSmallest(TreeNode* root, int k);
    void traverse(TreeNode* root, int k);

    // 2.二叉搜索树转换为累加树
        // 给出二叉搜索树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），
        // 使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
    TreeNode* convertBST(TreeNode* root);
    void traverse2(TreeNode* root);

    // 3.验证二叉树的合法性
    // 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
    // 有效 二叉搜索树定义如下：
        // 节点的左子树只包含 小于 当前节点的数。
        // 节点的右子树只包含 大于 当前节点的数。
        // 所有左子树和右子树自身必须也是二叉搜索树。
    bool isValidBST(TreeNode* root);
    bool _isValidBST(TreeNode* root, TreeNode* min, TreeNode* max);

    // 4.在二叉搜索树中搜索元素
    TreeNode* searchBST(TreeNode* root, int target);

    // 5.二叉搜索树中的插入操作
    // 给定二叉搜索树（BST）的根节点 root 和要插入树中的值 value，将值插入二叉搜索树。
    // 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。
    TreeNode* insertIntoBST(TreeNode* root, int val);

    // 6.二叉搜索树中的删除操作
    // 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点
    // ，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
    TreeNode* deleteNode(TreeNode* root, int key);
    TreeNode* getMin(TreeNode* node);

    // 7.寻找二叉搜索树的最大值
    int findMax(TreeNode* root);

    // 8.寻找二叉搜索树的最小值
    int findMin(TreeNode* root);

};

class Solution {
public:
    // *************************二叉搜索树构造********************
    // 1.不同的二叉搜索树
        // 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n
        // 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

    // 备忘录,消除重叠子问题
    vector<vector<int>> memo;    
    int numTrees(int n);
    // 定义：闭区间 [lo, hi] 的数字能组成 count(lo, hi) 种 BST
    int count(int lo, int hi);

    // 2. 不同的二叉搜索树Ⅱ
        // 给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。
    vector<TreeNode*> generateTrees(int n);
    // 定义：构造闭区间 [lo, hi] 组成的 BST，并返回根节点
    vector<TreeNode*> build(int lo, int hi);

    // ***********************二叉搜索树后序篇***********************
    // 全局变量，记录 BST 最大节点之和
    int maxSum = 0;

    // 1.二叉搜索子树的最大键值和
    // 给你一棵以 root 为根的 二叉树 ，请你返回 任意 二叉搜索子树的最大键值和。
    int maxSumBST(TreeNode* root);
    // 计算以 root 为根的二叉树的最大值、最小值、节点和,返回值：
    // res[0] 记录以 root 为根的二叉树是否是 BST，若为 1 则说明是 BST，若为 0 则说明不是 BST；
    // res[1] 记录以 root 为根的二叉树所有节点中的最小值；
    // res[2] 记录以 root 为根的二叉树所有节点中的最大值；
    // res[3] 记录以 root 为根的二叉树所有节点值之和。
    vector<int> findMaxMinSum(TreeNode* root);
};

// Trie/前缀树/字典树
// 一句话总结：Trie 树就是多叉树结构的延伸，是一种针对字符串进行特殊优化的数据结构。Trie 树在处理字符串相关操作时有诸多优势，比如节省公共字符串前缀的内存空间、方便处理前缀操作、支持通配符匹配等。
// Trie树的基本结构：Trie 树本质上就是一棵从二叉树衍生出来的多叉树。
// Trie 树节点实现
template<typename V>
struct TrieNode {
    V val = NULL; // 存储键对应的值
    TrieNode<V>* children[256] = { NULL }; // 存储指向子节点的指针
};
// TrieNode 节点本身只存储 val 字段,并没有一个字段来存储字符，字符是通过子节点在父节点的 children 数组中的索引确定的。形象理解就是，Trie 树用「树枝」存储字符串（键），用「节点」存储字符串（键）对应的数据（值）。

#endif // BINARY_TREE_H
