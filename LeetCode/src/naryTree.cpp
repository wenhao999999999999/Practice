#include <bits/stdc++.h>
#include "../include/naryTree.h"
using namespace std;

//多叉树的结点类的方法的定义
//叶子节点构造函数实现
TreeNode::TreeNode(int value) : val(value) {}

//添加叶子结点函数实现
void TreeNode::addChild(TreeNode* child) {
    children.push_back(child);
}

//多叉树类的方法的定义
//构造函数
NaryTree::NaryTree(TreeNode* node) : root(node) {}

//内部递归遍历函数的定义
void NaryTree::traverseInside(TreeNode* node) {
    //base case
    if (!node) return;
    
    //前序遍历
    cout << node->val << endl;

    //递归遍历子节点
    for(auto child : node->children) {
        traverseInside(child);
    }
    //后续位置
}

//多叉树递归遍历外部接口
void NaryTree::traverse() {
    traverseInside(root);
}

// 多叉树的层序遍历
vector<vector<int>> NaryTree::levelOrder() {
    if (!root) return {};
    queue<TreeNode*> q;
    vector<vector<int>> result;

    q.push(root);
    int depth = 1;

    while (!q.empty()) {
        int sz = q.size();
        vector<int> level;

        for (int i = 0; i < sz; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);

            for (auto child : node->children) {
                q.push(child);
            }
        }
        depth++;
        result.push_back(level);
    }
    return result;
}


