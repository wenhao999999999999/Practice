#include <queue>
#include <stack>
#include <unordered_map>
using namespace std;

// 用队列实现栈
class MyStack {
private:
    // 内部队列数据结构
    queue<int> q;

    // 存放栈顶元素
    int top_emem = 0;
public:
    // 添加元素到栈顶
    void push(int x) {
        q.push(x);
        top_emem = x;
    }

    // 删除栈顶元素并返回
    int pop() {
        int sz = q.size();
        while (sz > 2) {
            q.push(q.front());
            q.pop();
            sz--;
        }

        // 倒数第二个元素是新栈顶
        top_emem = q.front();
        q.push(q.front());
        q.pop();

        // 倒数第一个元素是返回值
        int value = q.front();
        q.pop();

        return value;
    }

    // 返回栈顶元素
    int top() {
        return top_emem;
    }

    // 判断栈是否为空
    bool empty() {
        return q.empty();
    }
};

// 最小栈
class MinStack {
private:
    // 记录栈中的最小元素
    stack<int> minstk;
    // 存储栈中的元素
    stack<int> stk;
public:
    void push(int x) {
        stk.push(x);
        // 注意：最小栈为空时，直接将元素压入栈
        if (minstk.empty() || x <= minstk.top()) {
            minstk.push(x);
        }
    }

    void pop() {
        if (stk.top() == minstk.top()) {
            minstk.pop();
        }
        stk.pop();
    }

    int top() {
        return stk.top();
    }

    int getMin() {
        return minstk.top();
    }
    
};

// 最大频率栈
class FreqStack {
private:
    // 记录栈中元素的最大频率
    int maxFreq;

    // 栈中元素的频率映射
    unordered_map<int, int> valToFreq;

    // 栈中所有频率对应的元素列表的映射
    unordered_map<int, stack<int>> freqToVal;
public:

    //  将一个整数 val 压入栈顶。
    void push(int val) {
        // 记录该元素出现的频率
        valToFreq[val] = valToFreq.count(val) ? valToFreq[val] + 1 : 1;

        // 更新栈中元素的最大频率
        int freq = valToFreq[val];
        maxFreq = max(maxFreq, freq);

        // 更新频率到元素的映射
        freqToVal[freq].push(val);
    }       

    // 删除并返回堆栈中出现频率最高的元素。
    // 如果出现频率最高的元素不只一个，则移除并返回最接近栈顶的元素。
    int pop() {
        // 找到返回值
        int val = freqToVal[maxFreq].top();

        // 弹出最大频率对应的值
        freqToVal[maxFreq].pop();

        // 当前值对应的频率-1
        valToFreq[val]--;

        // 更新最大频率
        if (freqToVal[maxFreq].empty()) {
            maxFreq--;
        }

        return val;
    }
};
