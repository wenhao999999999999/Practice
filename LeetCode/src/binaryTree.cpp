//定义二叉树类的方法
#include "../include/binaryTree.h"
#include <queue>
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

//Node构造函数实现（注意作用域！）
BinaryTree::Node::Node(int x) : val(x), left(nullptr), right(nullptr), next(nullptr) {}

//二叉树类构造函数
BinaryTree::BinaryTree() : root(nullptr) {}

//通过一个数组，创建一棵二叉树
    //输入：数组
    //返回：无
void BinaryTree::buildBinaryTree1(vector<int>& values) {

    //数组为空，直接返回
    if (values.empty()) return;

    //数组的第一个元素是二叉树的根节点
    root = new Node(values[0]);

    //初始化一个队列，存储已经成为二叉树一部分的结点
    queue<Node*> q;

    //根节点入队列
    q.push(root);

    //数组的下一个元素的索引
    size_t i = 1;

    while (!q.empty() && i < values.size()) {

        Node* cur = q.front(); q.pop();

        //-1表示当前节点为空
        //先左子树
        if (values[i] != -1) {
            cur->left = new Node(values[i]);
            q.push(cur->left);
        }

        i++;

        //后右子树
        if (i < values.size() && values[i] != -1) {
            cur->right = new Node(values[i]);
            q.push(cur->right);
        }

        i++;
    }
}

//构造最大二叉树辅助函数
    //输入：数组，低索引，高索引
    //返回：最大二叉树的根节点
BinaryTree::Node* BinaryTree:: build1(vector<int>& nums, int lo, int hi) {
    //base case
    if (lo > hi) return nullptr;

    //找到数组中最大值和对应的索引
    int index = -1, maxVal = INT_MIN;
    for (int i = lo; i <= hi; i++) {
        if (maxVal < nums[i]) {
            maxVal = nums[i];
            index = i;
        }
    }

    //先构造根节点
    Node* root = new Node(maxVal);

    //递归的构造左右子树
    root->left = build1(nums, lo, index-1);
    root->right = build1(nums, index+1, hi);

    return root;
}

//输入一个数组，按照最大二叉树原则：
    //1.创建一个根节点，其值为 nums 中的最大值。
    //2.递归地在最大值左边的子数组前缀上构建左子树。
    //3.递归地在最大值右边的子数组后缀上构建右子树。          
    //输入：数组
    //返回：二叉树根节点
BinaryTree::Node* BinaryTree::constructMaximumBinaryTree(vector<int>& nums) {
    if (nums.empty()) return nullptr;
    return build1(nums, 0, nums.size()-1);
}

//根据前序遍历和中序遍历构造二叉树辅助函数
    //输入：前序/中序遍历数组，起始索引，末尾索引
    //返回：二叉树根节点
BinaryTree::Node* BinaryTree::build2(vector<int> preorder, int preStart, int preEnd,
                 vector<int> inorder, int inStart, int inEnd) {
     //base case
     if (preStart > preEnd) {
        return nullptr;
     }

     //root的值是preorder中的第一个元素
     int rootVal = preorder[preStart];

     //rootVal在inorder中的索引
     int index = valToIndex[rootVal];

     //左子树的长度
     int leftSize = index - inStart;

     //先构造根节点
     Node* root = new Node(rootVal);

     //递归构造左子树
     root->left = build2(preorder, preStart+1, preStart+leftSize, 
                        inorder, inStart, index-1);

     //递归构造右子树
     root->right = build2(preorder, preStart+leftSize+1, preEnd,
                         inorder, index+1, inEnd);

     return root;
}

BinaryTree::Node* BinaryTree::buildBinaryTree2(vector<int> preorder, vector<int>& inorder) {
    //建立inorder中值到索引的映射，可以减少循环的次数
    int sz = inorder.size();
    for (int i = 0; i < sz; i++) {
        valToIndex[inorder[i]] = i;
    }

    return build2(preorder, 0, preorder.size()-1, 
                  inorder, 0, inorder.size()-1);
}

//根据中序遍历和后序遍历构造二叉树辅助函数
    //输入：中序/后序遍历数组，起始索引，末尾索引
    //返回：二叉树根节点
BinaryTree::Node* BinaryTree::build3(vector<int>& inorder, int inStart, int inEnd,
                                     vector<int>& postorder, int postStart, int postEnd) {
    //base case
    if (postStart > postEnd) {
        return nullptr;
    }

    //root的值是postorder中最后一个元素
    int rootVal = postorder[postEnd];
    
    //rootVal在inorder中的索引
    int index = valToIndex[rootVal];

    //右子树的长度
    int rightSize = inEnd - index;

    //先构造根节点
    Node* root = new Node(rootVal);

    //递归构造左子树
    root->left = build3(inorder, inStart, index-1,
                        postorder, postStart, postEnd-rightSize-1);

    //递归构造右子树
    root->right = build3(inorder, index+1, inEnd,
                         postorder, postEnd-rightSize, postEnd-1);

    //返回根节点
    return root;

}

BinaryTree::Node* BinaryTree::buildBinaryTree3(vector<int>& inorder, vector<int>& postorder) {
    //建立inorder中值到索引的映射，可以减少循环的次数
    int sz = inorder.size();
    for (int i = 0; i < sz; i++) {
        valToIndex[inorder[i]] = i;
    }

    return build3(inorder, 0, inorder.size()-1,
                  postorder, 0, postorder.size()-1);
}

//根据前序遍历和后序遍历构造二叉树，辅助函数
    //输入：前/后序遍历，起始/结尾索引
    //返回：二叉树根节点
    //构造原则：
        //1.首先把前序遍历结果的第一个元素确定为根节点的值。
        //2.然后把前序遍历结果的第二个元素作为左子树的根节点的值。
        //3.在后序遍历结果中寻找左子树根节点的值，从而确定了左子树的索引边界，进而确定右子树的索引边界，递归构造左右子树即可。
BinaryTree::Node* BinaryTree::build4(vector<int> preorder, int preStart, int preEnd,
                                     vector<int>& postorder, int postStart, int postEnd) {
    //base case
    if (preStart > preEnd) return nullptr;
    
    //当 preorder 只剩一个节点时，preorder[preStart + 1] 会越界
    if (preStart == preEnd) {
        return new Node(preorder[preStart]);
    }

    //root的值是preorder中第一个元素
    int rootVal = preorder[preStart];

    //root.left的值是前序遍历第二个元素
    int leftRootVal = preorder[preStart+1];
    
    //leftRootVal 在后序遍历数组中的索引
    int index = valToIndex[leftRootVal];

    //左子树的元素数目
    int leftSize = index - postStart + 1;

    //先构造出当前根节点
    Node* root = new Node(rootVal);

    // 递归构造左右子树
    // 根据左子树的根节点索引和元素个数推导左右子树的索引边界
    root->left = build4(preorder, preStart+1, preStart+leftSize,
                        postorder, postStart, index);
    // 这里 postEnd 应该是 postEnd - 1，因为 postorder[postEnd] 是当前根节点，不属于右子树
    root->right = build4(preorder, preStart+leftSize+1, preEnd,
                         postorder, index+1, postEnd-1);
    
    return root;
}

//根据前序遍历和后序遍历构造二叉树
    //输入：前/后序遍历
    //返回：二叉树根节点
BinaryTree::Node* BinaryTree:: buildBinaryTree4(vector<int>& preorder, vector<int>& postorder) {
    //建立postorder中值到索引的映射，可以减少循环的次数
    int sz = postorder.size();
    for (int i = 0; i < sz; i++) {
        valToIndex[postorder[i]] = i;
    }

    return build4(preorder, 0, preorder.size()-1,
                  postorder, 0, postorder.size()-1);
}





// 二叉树的层序遍历
    // 输入：无
    // 返回：二维数组
vector<vector<int>> BinaryTree::levelOrder() {

    // 存储返回的结果，数组的每行表示二叉树的一层
    vector<vector<int>> res;

    // 增强鲁棒性
    if (!root) return res;

    // 存储叶子节点的队列
    queue<Node*> q;

    // 根节点压入队列
    q.push(root);

    // 维护层数变量
    // 根节点在第一层
    int depth = 1;

    while (!q.empty()) {

        // 维护每一层节点数变量
        // 当前层中叶子节点的数量
        int sz = q.size();

        // 存储当前层中所有的叶子节点
        // 注意：level数组要放到while循环内部，每次清空数组
        vector<int> level;

        for (int i = 0; i < sz; i++) {

            // 出队列
            // 注意：在压入层数组前要将队首元素出队
            Node* node = q.front(); q.pop();

            cout << "depth = " << depth << " " << node->val << endl;

            // 压入当前层
            level.push_back(node->val);

            // 将当前叶子结点下一层的左右子树入队列
                //注意：子节点入队前需要先判断子树节点是否为空
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }

        // 一层遍历完后，压入结果数组
        res.push_back(level);

        // 层数加一
        depth++;
    }

    return res;
}

//二叉树的递归遍历
    //输入：无
    //返回：无
void BinaryTree::traverse(Node* root) {
    if (!root) return;
    //前序遍历
    // cout << root->val << endl;

    //前序遍历位置，进入结点，增加深度
    depth++;

    //前序遍历位置加入结点
    preordertraversal.push_back(root->val);

    //遍历到叶子结点时，记录最大深度
    if (root->left == nullptr && root->right == nullptr) {
        maxdepth = max(depth, maxdepth);
    }

    traverse(root->left);
    //中序遍历
    // cout << root->val << endl;
    traverse(root->right);
    //后序遍历
    // cout << root->val << endl;

    //后续位置，离开结点，减小深度
    depth--;
}

void BinaryTree::traverse() {
    traverse(root); // 调用私有递归方法
}

//统计二叉树结点数目，分解思路
    //输入：二叉树根节点
    //返回：二叉树节点数目
int BinaryTree::countInside(Node* root) {
    //base case
    if (!root) return 0;

    //计算左子树节点数目
    int leftreeNum = countInside(root->left);

    //计算右子树节点数目
    int rightreeNum = countInside(root->right);

    return leftreeNum + rightreeNum + 1;
}

int BinaryTree::count() {
    return countInside(root);
}

//返回二叉树的最大深度:分治问题的思路解法
//递归函数定义：
    //输入：二叉树根节点
    //返回：二叉树的最大深度
int BinaryTree::maxDepth1(Node* head) {
    //base case
    if (!head) return 0;

    //计算左子树的最大深度
    int lefMaxDepth = maxDepth1(head->left);

    //计算右子树的最大深度
    int righMaxDepth = maxDepth1(head->right);

    //计算并更新二叉树的最大直径
    int cur = lefMaxDepth + righMaxDepth;
    maxdiameter = max(maxdiameter, cur);

    //最大深度 = 左/右子树最大深度 + 根节点
    return max(lefMaxDepth, righMaxDepth) + 1;
}

//返回二叉树的最大深度：遍历的思路
int BinaryTree::maxDepth2(Node* root) {
    traverse(root);
    return maxdepth;
}


//返回二叉树的最大深度，外部调用
int BinaryTree::maxDepth() {
    // return maxDepth1(root);
    return maxDepth2(root);
}

//返回二叉树的前序遍历结果,分解问题思路
//递归函数的定义：
    //输入：二叉树根节点
    //返回：二叉树的前序遍历结果
vector<int> BinaryTree::preOrderTraversal(Node* root) {
    //初始化返回数组
    vector<int> res;

    //base case
    if (root == nullptr) return res;

    //根节点压入返回数组
    res.push_back(root->val);

    //左子树前序遍历结果
    vector<int> leftPre = preOrderTraversal(root->left);

    //右子树前序遍历结果
    vector<int> rightPre = preOrderTraversal(root->right);

    //合并结果
    res.insert(res.end(), leftPre.begin(), leftPre.end());
    res.insert(res.end(), rightPre.begin(), rightPre.end());

    //返回结果
    return res;
}

//返回二叉树前序遍历结果，外部调用接口
vector<int> BinaryTree::PreOrderTraversal() {
    //思路一：遍历
    // traverse(root);
    // return preordertraversal;

    //思路二：分解
    return preOrderTraversal(root);
}

//计算二叉树的直径
    //输入：二叉树根节点
    //返回：二叉树直径
int BinaryTree::diameterOfBinaryTree(Node* root) {
    maxDepth1(root);
    return maxdiameter;
}

//外部接口
int BinaryTree:: diameterOfBinaryTree() {
    return diameterOfBinaryTree(root);
}

//三叉树遍历
    //输入：二叉树的两个节点
    //返回：无
void BinaryTree::TernaryTreeTraversal(Node* node1, Node* node2) {
    //base case
    if (!node1 || !node2) return;

    // 将传入的两个结点连起来
    node1->next = node2;

    // 连接相同父结点的两个结点
    TernaryTreeTraversal(node1->left, node1->right);
    TernaryTreeTraversal(node2->left, node2->right);

    // 连接相邻结点
    TernaryTreeTraversal(node1->right, node2->left);
}

// 连接相邻结点的定义
    // 输入：二叉树根节点
    // 返回：二叉树根节点
BinaryTree::Node* BinaryTree:: connect(Node* root) {
    if (!root) return root;

    //遍历三叉树，连接相邻结点
    TernaryTreeTraversal(root->left, root->right);

    return root;
}

// 二叉树展开成链表,分解思路
    // 输入：二叉树根节点
    // 返回：无
    // 函数定义：输入二叉树根节点，二叉树展开成链表
void BinaryTree::flatten(Node* root) {
    // base case
    if (!root) return;

    // 先将左右子树各自展开成链表
    flatten(root->left);
    flatten(root->right);

    // 然后将展开的右子树接到左子树下方
    // 1.定义两条链表的头节点
    Node* leftList = root->left;
    Node* rightList = root->right;
    
    // 2.将左子树作为右子树
    root->right = leftList;
    root->left = nullptr;

    // 3.将原先右子树接到当前右子树末端
    Node* p = root;
    while(p->right) {    
        p = p->right;
    }
    p->right = rightList;
}

// 返回二叉树的序列化字符串
    // 输入：二叉树根节点
    // 返回：二叉树后序遍历字符串
string BinaryTree::serialize(Node* root) {
    // 空节点，用一个特殊字符表示
    if (!root) {
        return "#";
    }

    // 将左右子树序列化成字符串
    string left = serialize(root->left);
    string right = serialize(root->right);

    // 后序遍历代码位置
    // 左右子树加上自己，就是以自己为根的二叉树序列化结果
    string myself = left + "," + right + "," + to_string(root->val);

    //当前子二叉树序列出现的次数
    int freq = subTrees[myself];

    //多次重复也只会被加入结果集一次
    if (freq == 1) {
        duplicateSubtreesRoot.push_back(root);
    }

    // 给子树对应的出现次数加一
    subTrees[myself] = freq + 1;
    
    return myself;
}

// 寻找重复子树
    //输入：一棵二叉树的根节点 root
    //返回：所有重复的子树
    //注意：如果两棵树具有相同的结构和相同的结点值，则认为二者是重复的。
vector<BinaryTree::Node*> BinaryTree::findDuplicateSubtrees(Node* root) {
    serialize(root);
    return duplicateSubtreesRoot;
}


// 把一颗二叉树序列化成字符串,遍历方法实现
    // 输入：二叉树根节点
    // 返回：字符串
    // 方法：前序遍历
string Codec::serializeTraverse(TreeNode* root) {
    string sb;
    _serializeTraverse(root, sb);
    return sb;
}

 // 辅助函数，将二叉树存入 StringBuilder
void Codec::_serializeTraverse(TreeNode* root, string& sb) {
    if (!root) {
        sb.append(NULLSYM).append(SEP);
        //注意：空结点要返回
        return;
    }

    // 前序位置存入二叉树结点
    sb.append(to_string(root->val)).append(SEP);

    _serializeTraverse(root->left, sb);
    _serializeTraverse(root->right, sb);
}

// 把字符串反序列化成二叉树,遍历方法实现
    //输入：二叉树序列化后的字符串
    //返回：反序列化后的二叉树的根节点
TreeNode* Codec::deserializeTraverse(string data) {
    // 把字符串转化成列表
    list<string> nodes;

    // 实例化一个输入字符串流对象
    istringstream f(data);

    // 临时存储字符串
    string s;
    
    //注意：getline()函数参数的分隔符只能是字符
    while (getline(f, s, ',')) {
        nodes.push_back(s);
    }

    return _deserializeTraverse(nodes);

}
TreeNode* Codec::_deserializeTraverse(list<string>& nodes) {
    if (nodes.empty()) {
        return nullptr;
    }

    // ********前序位置********
    // 列表最左侧就是根节点
    string first = nodes.front();
    nodes.pop_front();

    // 如果为空，直接返回
    if (first == NULLSYM) return nullptr;

    // 如果不为空，创建一个结点
    TreeNode* root = new TreeNode(stoi(first));

    //递归构建左右子树
    root->left = _deserializeTraverse(nodes);
    root->right = _deserializeTraverse(nodes);

    return root;
}

// 二叉搜索树中第 K 小的元素
int BinarySearchTree::kthSmallest(TreeNode* root, int k) {
    traverse(root, k);
    return res;
}

void BinarySearchTree::traverse(TreeNode* root, int k) {
 // base case
 if (!root) return;

 traverse(root->left, k);

// **********中序位置***********
 rank++;

 if (rank == k) {
    res = root->val;
    // 注意：提前结束当前这层的递归函数
    return; 
 }

 traverse(root->right, k);

}

// 把二叉搜索树转换为累加数
TreeNode* BinarySearchTree::convertBST(TreeNode* root) {
    traverse2(root);
    return root;
}

void BinarySearchTree::traverse2(TreeNode* root) {
    // base case
    if (!root) return;

    // 先递归遍历右子树
    traverse2(root->right);

    // 注意：先求累加和再赋值
    // 维护累加和
    sum += root->val;
    
    // 将BST转化成累加树
    root->val = sum;

    // 后递归遍历左子树
    traverse2(root->left);
}

// 验证二叉搜索树
bool BinarySearchTree::isValidBST(TreeNode* root) {
    // 调用了辅助函数 _isValidBST，初始时没有上下界限制
    // （min = nullptr, max = nullptr），表示根节点可以是任意值
    return _isValidBST(root, nullptr, nullptr);
}

// 辅助函数
    // 输入：二叉树根节点，当前节点的取值范围
    // 返回：判断以当前节点为根节点的二叉树是否为二叉搜索树
///     
bool BinarySearchTree::_isValidBST(TreeNode* root, TreeNode* min, TreeNode* max) {
    // base case
    if (root == nullptr) return true; // 空节点是合法 BST。

    // 如果有最小限制 min，则当前节点必须严格大于 min->val
    if (min != nullptr && root->val <= min->val) return false;

    // 如果有最大限制 max，则当前节点必须严格小于 max->val
    if (max != nullptr && root->val >= max->val) return false;

    // 这保证了整个子树中的所有节点满足 BST 要求，而不仅仅是直接子节点


    // 根据定义，限定左子树的最大值是 root->val，右子树的最小值是 root->val
    // 递归的判断左右子树是否为二叉搜索树
    return _isValidBST(root->left, min, root) 
            && _isValidBST(root->right, root, max);    
}

// 在二叉搜索树中搜索元素
    // 输入：二叉搜索树根节点，目标值
    // 返回：目标值对应的二叉树节点
TreeNode* BinarySearchTree::searchBST(TreeNode* root, int target) {
    if (!root) return nullptr;

    // 当前节点的值大于目标值，左子树寻找
    if (root->val > target) return searchBST(root->left, target);

    // 当前节点的值小于目标值，右子树寻找
    if (root->val < target) return searchBST(root->right, target);

    // 当前节点的值等于目标值
    return root;
}

/// @brief 二叉搜索树中的插入操作
/// @param root 
/// @param val 
/// @return 新二叉搜索树根节点
TreeNode* BinarySearchTree::insertIntoBST(TreeNode* root, int val) {
    if (!root) {
        // 注意：二叉搜索树插入的一定是一个新的“叶子节点”。
        // 找到空位置插入新节点
        return new TreeNode(val);
    }

    // 插入节点的值大于当前节点的值，去右子树找插入位置
    if (root->val < val) {
        root->right = insertIntoBST(root->right, val);
    }

    // 插入节点的值小于当前节点的值，去左子树找插入位置
    if (root->val > val) {
        root->left = insertIntoBST(root->left, val);
    }

    // 返回 root，上层递归会接收返回值作为子节点
    return root;
}

/// @brief 删除二叉搜索树的指定节点
/// @param root 二叉搜索树根节点
/// @param key 要删除节点对应的值
/// @return 二叉搜索树根节点
TreeNode* BinarySearchTree::deleteNode(TreeNode* root, int key) {
    if (!root) return nullptr;

    if (root->val == key) {
        // 情况1：左右子节点为空
        // 情况2：左子树或右子树为空
        if (!root->left) return root->right;
        if (!root->right) return root->left;

        // 情况3：左右子树都不为空
        // 获取右子树中最小的元素
        TreeNode* minNode = getMin(root->right);

        // 删除右子树中最小的节点
        root->right = deleteNode(root->right, minNode->val);

        // 用右子树最小节点替换当前节点
        minNode->left = root->left;
        minNode->right = root->right;
        root = minNode;
        // 值小，左子树中寻找
    } else if (root->val > key) {
        root->left = deleteNode(root->left, key);
    } else if (root->val < key) {
        // 值大，右子树中寻找
        root->right = deleteNode(root->right, key);
    }

    return root;
}

/// @brief 获取二叉搜索树中的最小元素
/// @param root 二叉搜索树根节点
/// @return 最小值对应的节点
TreeNode* BinarySearchTree::getMin(TreeNode* root) {
    // BST 最左边的就是最小的
    while (root->left != nullptr) root = root->left;
    return root;
}

/// @brief 不同的二叉搜索树
/// @param n 整数 n ，求恰由 n 个节点组成且节点值从 1 到 n,互不相同的 二叉搜索树 有多少种？
/// @return 满足题意的二叉搜索树的种数。
    // 注意：二叉树算法的关键就在于明确根节点需要做什么
int Solution::numTrees(int n) {
    // 备忘录的值初始化为 0
    memo = vector<vector<int>>(n + 1, vector<int>(n + 1));
    return count(1,n);
}

/// @brief 闭区间 [lo, hi] 的数字能组成 count(lo, hi) 种 BST
/// @param lo 数字下界
/// @param hi 数字上界
/// @return 能组成的 BST 数目
int Solution::count(int lo, int hi) {
    // base case
        // 合法的 BST 为空，只有一种情况
    if (lo > hi) return 1;
    
    // 查备忘录
    if (memo[lo][hi] != 0) {
            return memo[lo][hi];
    }    

    int res = 0;

    // 循环统计造每个值作为根节点值的 BST
    for (int i = lo; i <= hi; i++) {

        // i 的值作为根节点 root
        // 递归统计左子树数量
        int left = count(lo, i - 1);

        // 递归统计右子树数量
        int right = count(i + 1, hi);

        res += left * right;
    }

    // 将结果存入备忘录
    memo[lo][hi] = res;

    return res;
}


/// @brief 不同的二叉搜索树Ⅱ
/// @param n 整数 n ，求恰由 n 个节点组成且节点值从 1 到 n,互不相同的 二叉搜索树
/// @return 满足题意的二叉搜索树
vector<TreeNode*> Solution::generateTrees(int n) {
    if (n == 0) return vector<TreeNode*>{};
    // 构造闭区间 [1, n] 组成的 BST 
    return build(1, n);    

}

/// @brief 构造闭区间 [lo, hi] 组成的 BST
/// @param lo 数字区间下界
/// @param hi 数字区间上界
/// @return 能组成的 BST 的根节点

// 递归函数定义：构造闭区间 [lo, hi] 组成的 BST，并返回 BST 的根节点数组
vector<TreeNode*> Solution::build(int lo, int hi) {
    vector<TreeNode*> res;
    // base case
         if (lo > hi) {
            // 这里需要装一个 null 元素，这样才能让下面的两个内层 for 循环都能进入，正确地创建出叶子节点
            // 举例来说吧，什么时候会进到这个 if 语句？当你创建叶子节点的时候，对吧。
            // 那么如果你这里不加 null，直接返回空列表，那么下面的内层两个 for 循环都无法进入
            // 你的那个叶子节点就没有创建出来，看到了吗？所以这里要加一个 null，确保下面能把叶子节点做出来
            res.emplace_back(nullptr);
            return res;
        } 
    
    // 1、穷举 root 节点的所有可能。
    for (int i = lo; i <= hi; i++) {
        // 2、递归构造出左右子树的所有有效 BST。
        vector<TreeNode*> leftTree = build(lo, i - 1);
        vector<TreeNode*> rightTree = build(i + 1, hi);
         
        // 3、给 root 节点穷举所有左右子树的组合。
        for (auto left : leftTree) {
            for (auto right : rightTree) {
                TreeNode* root = new TreeNode(i);
                root->left = left;
                root->right = right;
                res.emplace_back(root);                
            }
        }
    }
    return res;
}

// 返回二叉搜索树的最大值
int BinarySearchTree::findMax(TreeNode* root) {
    while(root->right != nullptr) root = root->right;
    return root->val; 
}

// 返回二叉搜索树的最小值
int BinarySearchTree::findMin(TreeNode* root) {
    while(root->left != nullptr) root = root->left;
    return root->val;
}


/// @brief 给你一棵以 root 为根的 二叉树 ，请你返回 任意 二叉搜索子树的最大键值和。
/// @param root 
/// @return 二叉搜索子树的最大键值和
/// 每个节点要做的事情：
///     1、左右子树是否是 BST。
///     2、左子树的最大值和右子树的最小值。
///     3、左右子树的节点值之和。         
int Solution::maxSumBST(TreeNode* root) {
    findMaxMinSum(root);
    return maxSum;
}

// 计算以 root 为根的二叉树的最大值、最小值、节点和
vector<int> Solution::findMaxMinSum(TreeNode* root) {
    // base case
    if (root == nullptr) {
        return vector<int>{1, INT_MAX, INT_MIN, 0};
    }

    // 递归计算左右子树
    vector<int> left = findMaxMinSum(root->left);  
    vector<int> right = findMaxMinSum(root->right);  
    
    // ******* 后序遍历位置 *******
    // 通过 left 和 right 推导返回值
    // 并且正确更新 maxSum 变量    
    vector<int> res(4);
    // 这个 if 在判断以 root 为根的二叉树是不是 BST
    if (left[0] == 1 && right[0] == 1 &&
        root->val > left[2] && root->val < right[1]) {
            // 以 root 为根的二叉树是 BST
            res[0] = 1;
            // 计算以 root 为根的这棵 BST 的最小值
            res[1] = min(left[1], root->val);
            // 计算以 root 为根的这棵 BST 的最大值
            res[2] = max(right[2], root->val);
            // 计算以 root 为根的这棵 BST 所有节点之和
            res[3] = left[3] + right[3] + root->val;
            // 更新全局变量
            maxSum = max(maxSum, res[3]);
        } else {
            // 以 root 为根的二叉树不是 BST
            res[0] = 0;
            // 其他的值都没必要计算了，因为用不到
        } 
        
        return res;
}







