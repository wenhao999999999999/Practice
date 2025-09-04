#include <bits/stdc++.h>
using namespace std;

//定义链表结点
struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* nxt) : val(x), next(nxt) {}
};

//声明链表类
class LinkedList {
private:
    ListNode* head;
    ListNode* tail;
public:
    LinkedList() : head(nullptr), tail(nullptr) {}

    //在尾部添加结点
    void addLast(int value) {
        ListNode* newNode = new ListNode(value);
        if (!head) {
            head = newNode;
            tail = newNode;
        } else {
            tail->next = newNode;
            tail = newNode;
        }
    }

    //单链表递归遍历
    void traverse(ListNode* head) {
        if (head == nullptr) return;

        //正序输出
        // cout << head->val << endl;

        //递归遍历
        traverse(head->next);

        //倒序输出
        cout << head->val << endl;
    }
};

