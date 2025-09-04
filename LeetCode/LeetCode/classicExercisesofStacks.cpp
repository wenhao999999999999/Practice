// 力扣中栈的经典习题
#include <bits/stdc++.h>
#include "../include/linkedList.h"
using namespace std;

class Solution {
public:
    // 1.简化路径，给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为 更加简洁的规范路径。
    // 在 Unix 风格的文件系统中规则如下：
        // 一个点 '.' 表示当前目录本身。
        // 此外，两个点 '..' 表示将目录切换到上一级（指向父目录）。
        // 任意多个连续的斜杠（即，'//' 或 '///'）都被视为单个斜杠 '/'。
        // 任何其他格式的点（例如，'...' 或 '....'）均被视为有效的文件/目录名称。
    // 返回的 简化路径 必须遵循下述格式：
        // 始终以斜杠 '/' 开头。
        // 两个目录名之间必须只有一个斜杠 '/' 。
        // 最后一个目录名（如果存在）不能 以 '/' 结尾。
        // 此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 '.' 或 '..'）。
        // 返回简化后得到的 规范路径 。

    string simplifyPath(string path) {
        vector<string> parts;
 
        // 实例化一个输入字符串流，用于解析有效路径
        istringstream ss(path);

        // 临时存储当前目录
        string part;

        // 借助栈解析最终的文件夹路径
        while (getline(ss, part, '/')) {
            // 当前目录本身或者为空，跳过
            if (part.empty() || part == ".") {
                continue;
            }
            // 目录切换到上一级
            if (part == "..") {
                // 目前已经解析的路径非空
                if (!parts.empty()) {
                    // 移除已经解析的最后一个目录
                    parts.pop_back();
                }
            }else {
                // 保存当前目录到结果路径中
                parts.push_back(part);
            }
        }

        // 栈中存储的文件夹组成路径
        string res;

        // 注意不可以改变原始路径
        for (const auto& p : parts) {
            res += "/" + p;
        }
        
        return res.empty() ? "/" : res;
    }

    // 2.重排链表
    // 给定一个单链表 L 的头节点 head ，单链表 L 表示为：
        // L0 → L1 → … → Ln - 1 → Ln
    // 请将其重新排列后变为：
        // L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
    // 不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
    void reorderList(ListNode* head) {
        // 初始化一个栈，用于存放链表的结点
        stack<ListNode*> stk;

        // 先把所有节点装进栈里，得到倒序结果
        ListNode* p = head;
        while (p != nullptr) {
            stk.push(p);
            p = p->next;
        }

        p = head;

         while (!stk.empty()) {
            // 链表尾部的节点
            ListNode* lastNode = stk.top();
            stk.pop();

            // 存储当前节点的下一个结点
            ListNode* next = p->next;

            if (lastNode == next || lastNode->next == next) {
                //偶数 || 奇数长度链表结束条件
                //注意：指针指向空
                lastNode->next = nullptr;
                break;
            }

            p->next = lastNode;
            lastNode->next = next;
            p = next;
         }    
    }

    // 3.有效的括号
    // 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
    // 有效字符串需满足：
        // 左括号必须用相同类型的右括号闭合。
        // 左括号必须以正确的顺序闭合。
        // 每个右括号都有一个对应的相同类型的左括号。
    bool isValid(string s) {
        // 存储左括号的栈
        stack<char> left;

        // 遍历字符串中的括号
        for (char c : s) {

            // 如果是左括号就入栈
             if (c == '(' || c == '{' || c == '[') {
                left.push(c);

            // 字符 c 是右括号
             } else {
                if (!left.empty() && leftOf(c) == left.top()) {
                    left.pop();
                } else {

                    // 和最近的左括号不匹配
                    return false;
                }
             }
        }
        // 是否所有的左括号都被匹配了
        // 注意：这里不是直接返回true
        return left.empty();
    } 
    
    char leftOf(char c) {
        if (c == '}') return '{';
        if (c == ')') return '(';
        return '[';
    }

    // 4.逆波兰表达式求值
        // 给你一个字符串数组 tokens ，表示一个根据逆波兰表示法表示的算术表达式。
        // 请你计算该表达式。返回一个表示表达式值的整数。
        // 解题思路：按顺序遍历逆波兰表达式中的字符，如果是数字，则放入栈；如果是运算符，则将栈顶的两个元素拿出来进行运算，再将结果放入栈。对于减法和除法，运算顺序别搞反了，栈顶第二个数是被除（减）数。
    int evalRPN(vector<string>& tokens) {
        // 初始化一个栈，用于存放操作数
        stack<int> stk;

        // 遍历字符串
        for (const std::string& token : tokens) {

            //如果是运算符
            if (token == "+" || token == "-" || token == "*" || token == "/") {

                //从栈顶拿出两个数字进行运算，运算结果入栈
                int a = stk.top(); stk.pop();
                int b = stk.top(); stk.pop(); 
                
                if (token == "+") stk.push(b + a);
                else if (token == "*") stk.push(b * a);
                else if (token == "-") {
                    // 对于减法和除法，顺序别搞反了，第二个数是被除（减）数
                    stk.push(b - a);
                }
                else if (token == "/") stk.push(b / a);                
            } else {
                //如果是个数字，直接入栈即可
                stk.push(stoi(token));
            }
        }
        
        // 最后栈中剩下一个数字，即是计算结果
         return stk.top();
    }

    // 5.文件的最长绝对路径
    int lengthLongestPath(string input) {
        
        // 存储当前路径中每一层目录/文件名的长度（不含 /）。
        deque<int> stack;
        // 记录最长路径长度
        int maxLen = 0;

        // 把原始字符串按 '\n' 拆成多行，每一行一个目录或文件。
        vector<string> parts = split(input, '\n');
        
        for (const string& part : parts) {
            // level 是当前行的深度（\t 的数量），用 rfind("\t") + 1 得到
            int level = part.rfind("\t") + 1;

            // 如果当前目录层级比栈中的层级低，弹出多余层级，使栈保持正确的父路径结构。
            while (level < stack.size()) {
                stack.pop_back();
            }

            // 当前这层目录/文件的实际名字长度（去掉 \t）
            stack.push_back(part.size() - level);

            // 如果是文件，就计算路径长度
            if (part.find('.') != string::npos) {
                int sum = 0;
                for (int len : stack) {
                    sum += len;
                }
                sum += stack.size() - 1;  // 添加 `/` 分隔符数量
                maxLen = max(maxLen, sum);
            }
        }
        return maxLen;
    }

    // 工具函数：按分隔符切分字符串
    vector<string> split(const string& s, char delimiter) {
        vector<string> tokens;
        string token;
        istringstream tokenStream(s);
        while (getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }
};

// ************* 单调栈算法 **************** \\
// 单调栈用途不太广泛，只处理一类典型的问题，比如「下一个更大元素」，「上一个更小元素」等。

/// @brief 输入一个数组 nums，请你返回一个等长的结果数组，结果数组中对应索引存储着下一个更大元素，如果没有更大的元素，就存 -1。
/// @param nums 
/// @return res
vector<int> calculateGreaterElement(vector<int>& nums) {
    int sz = nums.size();

    vector<int> res(sz);

    stack<int> stk;

    // 倒着往栈里放
    for (int i = sz - 1; i >= 0; i--) {
        // 判定各自高矮
        while (!stk.empty() && stk.top() <= nums[i]) {
            // 矮个起开，反正也被挡着了。。。
            stk.pop();
        }

        res[i] = stk.empty() ? -1 : stk.top();
        stk.push(nums[i]);
    }

    return res;
}
// for 循环要从后往前扫描元素，因为我们借助的是栈的结构，倒着入栈，其实是正着出栈。


/// 问题变形
// 1.下一个更大元素Ⅰ
// nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。
// 给你两个 没有重复元素 的数组 nums1 和 nums2 ，下标从 0 开始计数，其中nums1 是 nums2 的子集。
// 对于每个 0 <= i < nums1.length ，找出满足 nums1[i] == nums2[j] 的下标 j ，并且在 nums2 确定 nums2[j] 的 下一个更大元素 。如果不存在下一个更大元素，那么本次查询的答案是 -1 。

/// @brief 因为题目说 nums1 是 nums2 的子集，那么我们先把 nums2 中每个元素的下一个更大元素算出来存到一个映射里，然后再让 nums1 中的元素去查表即可：
/// @param nums1 
/// @param nums2 
/// @return 
vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
    vector<int> greater = calculateGreaterElement(nums2);

    unordered_map<int, int> greaterMap;
    
    for (int i = 0; i < nums2.size(); i++) {
        greaterMap[nums2[i]] = greater[i];
    }

    vector<int> res(nums1.size());

    for (int i = 0; i < nums1.size(); i++) {
        res[i] = greaterMap[nums1[i]];
    }

    return res;
}

// 2.每日温度
// 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> res(n);
    stack<int> index;

    for (int i = n - 1; i >= 0; i--) {
        while (!index.empty() && temperatures[i] >= temperatures[index.top()]) {
            index.pop();
        }

        res[i] = index.empty() ? 0 : index.top() - i;
        index.push(i);
    }
    return res;

}

// 括号类问题汇总
class Solution {
public:

    // 1.有效的括号
        // 输入：一个字符串
        // 返回：字符串是否有效
    bool isValid(string s) {
        // 初始化一个栈，用于存储左括号
        stack<char> stk;

        for (auto c : s) {
            if (c == '('|| c == '[' || c == '{') {
                stk.push(c);
            // 注意：要先判断栈是否为空，否则会报错
            } else if (!stk.empty() && leftOf(c) == stk.top()) {
                stk.pop();
            } else {
                return false;
            }
        }
        return stk.empty();
    }

    // 辅助函数：根据右括号类型并返回适配的的左括号
    char leftOf(char c) {
        if (c == ')') return '(';
        if (c == ']') return '[';
        return '{';
    }

    // 2.使括号有效的最少添加
        // 输入：字符串 s
        // 返回：为使结果字符串 s 有效而必须添加的最少括号数
    int minAddToMakeValid(string s) {
        // 右括号的需求数量
        int need = 0;

        // 左括号的需求数量
        int res = 0;

        int sz = s.size();
        for (int i = 0; i < sz; i++) {

            // 如果当前字符时左括号，就需要一个右括号
            if (s[i] == '(') {
                need++;
            } else {
                // 如果当前字符是右括号，右括号的需求就可以满足1个
                need--;
                // 如果目前的右括号多出来了，没有与他匹配的左括号
                if (need == -1) {
                    // 需要插入一个左括号；
                    res++;
                    // 正好匹配，不需要插入右括号了
                    need = 0;
                }
            }
        }
        // 返回需要插入的左右括号总数
        return res + need;
    } 
    
    // 3.平衡括号字符串的最少插入次数
        // 输入：字符串 s
        // 返回：让 s 平衡的最少插入次数
        // 一个括号字符串被称为平衡的当它满足：
            // 任何左括号 '(' 必须对应两个连续的右括号 '))' 。
            // 左括号 '(' 必须在对应的连续两个右括号 '))' 之前。
            // 比方说 "())"， "())(())))" 和 "(())())))" 都是平衡的， ")()"， "()))" 和 "(()))" 都是不平衡的。
        // 你可以在任意位置插入字符 '(' 和 ')' 使字符串平衡。
    int minInsertions(string s) {
        // need 记录右括号的需求量
        int need = 0;

        // 记录需要插入的括号总数
        int res = 0;

        for (int i = 0; i < s.length(); i++) {
            // 一个左括号对应两个右括号
            if (s[i] == '(') {
                need += 2;

                // 如果当前需要的右括号数是奇数，说明前面出问题了
                if (need % 2 == 1) {
                    // 插入一个右括号来补全之前的需求
                    res++;

                    //减少一个需求（让 need 从奇数变成偶数）
                    need--; 
                }
            }

            if (s[i] == ')') {
                need--;
                if (need == -1) {
                    // 需要插入一个左括号
                    res++;
                    // 一个左括号对应2个右括号，但现在多出来1个，还需要1个
                    need = 1;
                }
            }
        }

        // res 是过程中需要插入的括号数（包括多插的 ')' 或 '('）
        // need 是最后还差的 ')' 数量（即闭合用的右括号）
        return res + need;
    }
};