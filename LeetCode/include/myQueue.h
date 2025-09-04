// 用两个栈实现队列
#include <stack>
using namespace std;

class MyQueue {
private:
    // 内部数据结构，两个栈
    stack<int> stk1, stk2;

public:
    MyQueue() {

    }

    // 队尾添加元素
    void push(int x) {
        stk1.push(x);
    }

    // 返回队首元素
    int peek() {
        if (stk2.empty()) {
            while (!stk1.empty()) {
                stk2.push(stk1.top());
                stk1.pop();
            }
        }
        return stk2.top();
    }

    // 移除队首元素并返回
    int pop() {
        //先调用peek()保证stk2非空
        peek();

        int poppedValue = stk2.top();
        stk2.pop();
        return poppedValue;
    }

    // 判断栈是否为空
    bool empty() {
        return stk1.empty() && stk2.empty();
    }
};
