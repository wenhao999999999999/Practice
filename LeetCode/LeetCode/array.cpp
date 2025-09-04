#include <bits/stdc++.h>
using namespace std;
// 数组快慢指针问题

// 1.删除有序数组中的重复项
// 给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

class Solution {
public:
    // 解题思路：定义两个指针和一个计数变量，维护nums[0..i]元素出现次数不超过 2 次
    int removeDuplicates(vector<int>& nums) {
        if (nums.size() <= 2) return nums.size();

        int slow = 0;
        int fast = 0;
        int count = 0;

        while (fast < nums.size()) {
            // 如果新元素与当前元素不相等可以直接赋值
            if (nums[fast] != nums[slow]) {
                // 注意：先+1，再赋值
                slow++;
                nums[slow] = nums[fast];
            // 注意：这里的判断条件是slow < fast 
            } else if (slow < fast && count < 2) {
                // 注意：先+1，再赋值
                slow++;
                nums[slow] = nums[fast];
            }

            // 快指针向前移动
            fast++;
            count++;
            // 注意：这里的
            if (fast < nums.size() && nums[fast] != nums[fast - 1]) {
                // fast 遇到新的不同元素，重置 count
                count = 0;
            }
        }
        return slow + 1;
    }
};

// 2.轮转数字
// 给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
// 原地修改：第一次反转全部元素；第二次反转前 k 个元素；第三次反转其余元素
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k %= n; // 处理 k 大于数组长度的情况，取余后得到实际需要轮转的步数
        // 第一次反转整个数组
        reverse(nums.begin(), nums.end());
        // 第二次反转前 k 个元素
        reverse(nums.begin(), nums.begin() + k);
        // 第三次反转剩下的元素
        reverse(nums.begin() + k, nums.end());
    }
};


// 3.多数元素
// 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。你可以假设数组是非空的，并且给定的数组总是存在多数元素。

// 哈希表解法：用一个哈希表记录数组中每个元素出现的次数
class Solution {
private:
    // 初始化一个哈希表，统计数组中每个元素出现的次数
    unordered_map<int,int> mp;
public:
    int majorityElement(vector<int>& nums) {
        int res = 0;
        int count = 0;
        for (auto num : nums) {
            mp[num]++;
        }

        for(auto num : nums) {
            if (count < mp[num]){
                res = num;
                count = mp[num];
            }
                
        }

        return res;
    }
};

// 摩尔投票解法：
// 维护一个候选多数元素和一个计数器。遍历数组时，遇到相同的元素则计数器加 1，遇到不同的元素则计数器减 1。当计数器为 0 时，更换候选元素为当前元素，并将计数器重置为 1。最终，候选元素即为多数元素（因为题目保证数组中一定存在多数元素）。

class Solution {
public:
    int majorityElement(vector<int>& nums) {
        // 计数器
        int count = 0;

        // 候选元素
        int target = 0;

        for (int i  = 0; i < nums.size(); i++) {
            if (count == 0) {
                target = nums[i];
                count = 1;
            }else if (target == nums[i]) {
                count++;
            } else {
                count--;
            }
        }

        return target;
    }
};

// 最大子数组和问题/Kadane 算法


// 4. H 指数
// 题干：
    // 给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数。计算并返回该研究者的 h 指数。根据维基百科上 h 指数的定义：h 代表“高引用次数” ，一名科研人员的 h 指数 是指他（她）至少发表了 h 篇论文，并且 至少 有 h 篇论文被引用次数大于等于 h 。如果 h 有多种可能的值，h 指数 是其中最大的那个。
// 思路：
    // 排序：把 citations 按照降序排序。

    // 遍历：依次检查第 i 篇论文的引用次数是否 ≥ i+1。

        // 例如：排序后数组 [6,5,3,1,0]

        // 第 1 篇论文 6 ≥ 1 → h ≥ 1

        // 第 2 篇论文 5 ≥ 2 → h ≥ 2

        // 第 3 篇论文 3 ≥ 3 → h ≥ 3

        // 第 4 篇论文 1 ≥ 4 ❌ → 停止

    // 答案：最大的满足条件的 i+1 就是 H 指数。
    // 排序后，第 i 篇论文如果至少被引用 i+1 次，说明前 i+1 篇论文都达标，可以更新 H 指数。直到遇到不满足的情况为止，最大值就是答案。
class Solution {
public:
    int hIndex(vector<int>& citations) {
        // 注意点 1 : sort() 函数默认升序排序
        sort(citations.begin(), citations.end(), greater<int>());
        int h = 0;
        // 遍历数组，依次检查第 i 篇论文的引用次数是否 ≥ i + 1
        for (int i = 0; i < citations.size(); i++) {
            if (citations[i] >= i+1) h++;
            else break;
        }
        return h;

    }
};

// ----------------------------- 二维数组 ------------------------------ 
// 1. 反转字符串中的单词
// 题干：
    // 给你一个字符串 s ，请你反转字符串中单词的顺序。单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的单词分隔开。返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。

// 示例 1：
    // 输入：s = "the sky is blue"
    // 输出："blue is sky the"

// 思路：先把整个字符串进行翻转，再把每个单词中的字母翻转。

// 代码：
class Solution {
public:
    string reverseWords(string s) {
        istringstream iss(s);
        string word;
        vector<string> words;
        while(iss >> word){
            words.push_back(word);
        }
        reverse(words.begin(), words.end());
        string res;
        for(int i = 0; i < words.size(); i++){
            res += words[i];
            if(i != words.size() - 1) res += " ";
        }
        return res;
    }
};

// 2. 反转字符串中的单词 Ⅱ
// 题干：
    // 给你一个字符数组 s ，反转其中 单词 的顺序。单词 的定义为：单词是一个由非空格字符组成的序列。s 中的单词将会由单个空格分隔。必须设计并实现 原地 解法来解决此问题，即不分配额外的空间。
// 思路：
    // 先把整个字符数组反转；再把每个单词各自反转回来。因为题目保证单词之间只有一个空格、无首尾空格，这样即可把“单词顺序”反转且不分配额外空间。
// 代码：
class Solution {
public:
    void reverseWords(vector<char>& s) {
        int n = (int)s.size();
        // 1) 整体反转
        reverse(s.begin(), s.end());

        // 2) 逐个单词反转
        int i = 0;
        while (i < n) {
            int j = i;
            while (j < n && s[j] != ' ') ++j;   // [i, j) 是一个单词
            // 注意点1：reverse() 函数左闭右开
            // 注意点2：reverse() 函数的入参是迭代器
            reverse(s.begin() + i, s.begin() + j);
            i = j + 1; // 跳到下一个单词的开头
        }
    }
};
