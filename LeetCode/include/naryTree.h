#ifndef NARYTREE_H
#define NARYTREE_H

#include <vector>
using namespace std;

//叶子结点类声明
class TreeNode {
public:
    int val;
    vector<TreeNode*> children;

    //声明叶子结点构造函数
    TreeNode(int val);

    //声明子结点添加函数
    void addChild(TreeNode* child);
};

//多叉树的类的声明
class NaryTree {
private:

    
    //声明多叉树内部遍历函数
    void traverseInside(TreeNode* node);
public:
    //定义多叉树的根节点是私有变量
    TreeNode* root;
    //声明多叉树的构造函数
    NaryTree(TreeNode* _root);

    //声明多叉树的外部可调用的递归遍历函数
    void traverse();
    // 多叉树的层序遍历
    vector<vector<int>> levelOrder();

};

#endif // TREENODE_H
