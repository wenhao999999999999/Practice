#include <bits/stdc++.h>
using namespace std;
// 滑动窗口练习

// 滑动窗口可以归为快慢双指针，一快一慢两个指针前后相随，中间的部分就是窗口。滑动窗口算法技巧主要用来解决子数组问题，比如让你寻找符合某个条件的最长/最短子数组。


// 滑动窗口解题框架
// int slidwindow (vector<int>& nums){
//     int left = 0;
//     int right = 0;

//     while (right < nums.size()) {
//         // 增大窗口
//         window.addLast(nums[right]);
//         right++;

//         while (window need to shrink) {
//             // 缩小窗口
//             window.removeFirst(nums[left]);
//             left++:
//         }
//     }
// }
// 基于滑动窗口算法框架写出的代码，时间复杂度是 O(N)


// 滑动窗口伪代码框架

// int slidingWindow(string s) {
//     // 用合适的数据结构记录窗口中的数据，根据具体场景变通
//     // 比如说，我想记录窗口中元素出现的次数，就用 map
//     // 如果我想记录窗口中的元素和，就可以只用一个 int
//     // auto window = ...;

//     int left = 0, right = 0;
//     while (right < s.size()) {
//         // c 是将移入窗口的字符
//         char c = s[right];
//         window.add(c);
//         right++;

//         // 进行窗口内数据的一系列更新
//         // ...

//         // *** debug 输出的位置 ***
//         // printf("window: [%d, %d)\n", left, right);

//         while(window needs shrink) {
//             // // d 是将移出窗口的字符
//             char d = s[left];
//             window.remove(d);
//             left++;

//             // 进行窗口内数据的一系列更新
//             // ...
//         }

//     }
// }

// 基于这个框架，遇到子串/子数组相关的题目，你只需要回答以下三个问题：
// 1、什么时候应该移动 right 扩大窗口？窗口加入字符时，应该更新哪些数据？
// 2、什么时候窗口应该暂停扩大，开始移动 left 缩小窗口？从窗口移出字符时，应该更新哪些数据？
// 3、什么时候应该更新结果？

// 练习题
// 1.长度最小的子数组
// 给定一个含有 n 个正整数的数组和一个正整数 target 。找出该数组中满足其总和大于等于 target 的长度最小的 子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

class Solution {
public:
    // 算法流程：定义两个指针，维护一个窗口；定义 sum 变量，记录窗口 [left, right) 内元素的和；扩大窗口条件： right < nums.size()，更新 sum；缩小窗口条件：sum >= target;定义一个 res ,维护结果；如果 sum >= target , 更新res;
    int minSubArrayLen(int target, vector<int>& nums) {
        int left = 0, right = 0;
        int sum = 0;
        int res = INT_MAX;

        // 扩大窗口条件
        while (right < nums.size()) {
            // 更新窗口和数据
            sum += nums[right];
            // 更新右指针
            right++;

            // 缩小窗口条件
            while (sum >= target) {
                // 更新结果
                res = min(res, right - left);

                // 更新和
                sum -= nums[left];
                left++;
            }
        }

        // 注意：当所有元素和都小于 target时，要返回 0；

        return res == INT_MAX ? 0 : res;
    }
};

// 2. 串联所有单词的子串
// 题干：
    // 给定一个字符串 s 和一个字符串数组 words。 words 中所有字符串 长度相同。
    //  s 中的 串联子串 是指一个包含  words 中所有字符串以任意顺序排列连接起来的子串。
    // 例如，如果 words = ["ab","cd","ef"]， 那么 "abcdef"， "abefcd"，"cdabef"， "cdefab"，"efabcd"， 和 "efcdab" 都是串联子串。 "acdbef" 不是串联子串，因为他不是任何 words 排列的连接。
    // 返回所有串联子串在 s 中的开始索引。你可以以 任意顺序 返回答案。
// 思路：
    // 1. 由于 words 中的每个字符串长度相同，首先可以根据 words 中单词的长度计算出目标子串的长度。

    // 2. 然后滑动窗口遍历字符串 s，每次截取一个子串并判断它是否是 words 中各个字符串的拼接。

    // 3. 使用哈希表来存储 words 中每个单词出现的次数，滑动窗口内的子串的每个单词的次数也要匹配。
// 代码：
class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        vector<int> result;
        if (words.empty() || s.empty()) return result; // 如果 words 或 s 为空，直接返回空结果。

        // ========= 初始化 ===========
        int word_len = words[0].size(); // 每个单词长度相同，记为 word_len。
        int num_words = words.size(); // 单词个数为 num_words
        int total_len = word_len * num_words; // 整个拼接串的总长度为 total_len = word_len * num_words。

        // ============= 构建单词计数哈希表 ============
        unordered_map<string, int> word_count;
        for (const string& word : words) {
            word_count[word]++; // 用哈希表统计每个单词在 words 中的出现次数。
        }

        // ================= 遍历所有可能的起点 ===============
        for (int i = 0; i <= s.size() - total_len; i++) {
            unordered_map<string, int> seen; // 用 seen 哈希表统计当前窗口里实际看到的单词次数。
            int j = 0;
            // 循环 j = 0 ~ num_words-1
            while (j < num_words) {
                int start = i + j * word_len;
                if (start + word_len > s.size()) break;  // 保证不越界

                string word = s.substr(start, word_len); // 逐个取出窗口内的单词

                // 检查单词合法性
                if (word_count.find(word) != word_count.end()) {
                    // 如果 word 在 word_count 中：更新 seen[word]
                    seen[word]++;
                    // 如果次数超过 word_count[word] → 不合法，退出。
                    if (seen[word] > word_count[word]) break; 
                } else {
                    // 如果 word 不在 words 中 → 直接退出。
                    break; 
                }
                j++;
            }
            
            // 判断是否完全匹配
            if (j == num_words) {
                result.push_back(i); // 如果成功匹配了所有单词（j == num_words），说明以 i 为起点的子串符合条件 → 记录 i。
            }
        }

        return result;
    }
};