#include <bits/stdc++.h>
using namespace std;
// 哈希表练习题目

// 1.最长连续序列
// 题目描述：
    // 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。请你设计并实现时间复杂度为 O(n) 的算法解决此问题。    
// 解题思路：
    // 将数组转换为哈希集合，快速判断某个元素是否存在并自动去重；遍历集合，找到连续序列的起点；计算以起点开始的连续序列的长度；返回结果 
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> set(nums.begin(), nums.end());
        // 存储结果
        // 注意：结果应该初始化为0，而不是INT_MIN，对应空数组的情况
        int res = 0;

        
        // 注意：这里要遍历集合而不是遍历数组
        for (auto num : set) {
            // 判断某个元素是否为序列起点
            if (set.find(num - 1) != set.end()) {
                // 判断当前元素前面是否有相邻元素，只有前面不存在相邻元素时，才能以当前元素为起点构建序列
                continue;
            }

            // 当前起点
            int curnum = num;
            // 初始长度
            int curlen = 1;

            while (set.find(curnum+1) != set.end()) {
                curnum++;
                curlen++;
            }

            res = max(res, curlen);
        }

        return res;
    }
};

// 2.赎金信
// 题目描述：
    // 给你两个字符串：ransomNote 和 magazine ，判断 ransomNote 能不能由 magazine 里面的字符构成。如果可以，返回 true ；否则返回 false 。magazine 中的每个字符只能在 ransomNote 中使用一次。
// 解题思路：
    // 用哈希表存储 magazine 中的字符出现的次数；遍历字符串 ransomNote ，判断哈希表中是否包含所有字符；
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        unordered_map<char, int> map; // 键：字符；值：字符出现的次数

        for (auto c : magazine) {
            map[c]++;
        }

        for (auto c : ransomNote) {
            if (!map.count(c)) return false;
            map[c]--;
            if (map[c] == 0) map.erase(c);
        }

        return true;
    }
};

// 3.O(1) 时间插入、删除和获取随机元素
// 题目描述：
    // 实现RandomizedSet 类：
    // RandomizedSet() 初始化 RandomizedSet 对象
    // bool insert(int val) 当元素 val 不存在时，向集合中插入该项，并返回 true ；否则，返回 false 。
    // bool remove(int val) 当元素 val 存在时，从集合中移除该项，并返回 true ；否则，返回 false 。
    // int getRandom() 随机返回现有集合中的一项（测试用例保证调用此方法时集合中至少存在一个元素）。每个元素应该有 相同的概率 被返回。
    // 你必须实现类的所有函数，并满足每个函数的平均时间复杂度为 O(1) 。

// 解题思路：
    // 用连续内存的数组数据结构存储类内的每个元素；用哈希表存储每个元素在数组中的索引；删除元素时把元素移动到数组末尾再删除；用随机数发生器生成数组索引随机数，返回数组中的元素；

// 核心代码实现：
class RandomizedSet {
private:
    vector<int> nums;
    unordered_map<int, int> index;
    default_random_engine random; // 声明一个名为random的随机数引擎对象
public:
    RandomizedSet() {
        random.seed(random_device()()); // 外层std::random_device()：创建临时真随机数设备对象;内层()：调用operator()生成真随机数
    }
    
    bool insert(int val) {
        // 若 val 已存在，不用再插入
        if (index.count(val)) {
            return false;
        }

        nums.push_back(val);
        index[val] = nums.size() - 1;
        return true;
    }
    
    bool remove(int val) {
        // 若 val 不存在，不用再删除
        if (!index.count(val)) {
            return false;
        }

        int i = index[val]; // 先拿到 val 的索引 i
        int lastnum = nums.back(); // 获取最后一个元素的值 lastnum
        index[lastnum] = i; // 将最后一个元素对应的索引修改为 i
        nums[i] = lastnum; // 交换 val 和最后一个元素
        nums.pop_back();
        index.erase(val);
        return true;
    }
    
    int getRandom() {
        uniform_int_distribution<int> dist(0, nums.size() - 1); // 均匀整数分布对象，它定义了随机数的范围（0 到 nums.size()-1）
        return nums[dist(random)];
        // 代码解释：
            // random() 产生一个很大的随机整数（如 0~4,294,967,295）
            // dist (分布器) 将引擎输出的原始随机数映射到指定范围并保证概率分布特性
    }
};

// 4.单词规律
// 题目描述：
    // 给定一种规律 pattern 和一个字符串 s ，判断 s 是否遵循相同的规律。这里的 遵循 指完全匹配，例如， pattern 里的每个字母和字符串 s 中的每个非空单词之间存在着双向连接的对应规律。
// 解题思路：
    //用哈希表存储键值对，pattern 中的元素充当键，s中对应位置（以空格作为间隔）的元素作为值；遍历字符串s，检查同一键对应的值是否相等。
// 核心代码：
class Solution {
public:
    bool wordPattern(string pattern, string s) {
        // 分割字符串
        vector<string> words;
        stringstream ss(s);
        string word;
        while (ss >> word) {
            words.push_back(word);
        }

        // 长度检查
        if (pattern.size() != words.size()) return false;

        // 建立键值对的双向映射关系
        unordered_map<char, string> charTostring;
        unordered_map<string, char> stringTochar;

        for (int i = 0; i < pattern.size(); i++) {
            char c = pattern[i];
            string w = words[i];

            // 检查 char -> string 映射
            if (charTostring.count(c)) {
                if (charTostring[c] != w) return false;
            }
             
            // 检查 string -> char 映射
            else if (stringTochar.count(w)) {
                if (stringTochar[w] != c) return false;
            }
            
            // 新建键值对映射
            else {
                charTostring[c] = w;
                stringTochar[w] = c;
            }
        }

        return true;

    }
};

// 5.快乐数
// 题目描述：
    // 编写一个算法来判断一个数 n 是不是快乐数。「快乐数」 定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果这个过程 结果为 1，那么这个数就是快乐数。如果 n 是 快乐数 就返回 true ；不是，则返回 false 。
// 解题思路：
    // 使用哈希表记录计算过程中出现的所有中间结果：用集合存储计算过程中出现的数字，重复计算数字的各位平方和，当计算结果为1时，返回true，当计算结果在集合中已存在时，说明进入循环，返回false
// 核心代码：
class Solution {
public:
    bool isHappy(int n) {
        unordered_set<int> set;
        set.insert(n);
        int num = getSquare(n);
        while (num != 1) {
            if (set.count(num)) return false;
            set.insert(num);
            num = getSquare(num);  
        }

        return true;
    }

    int getSquare (int n) {
        int sum = 0;
        while (n > 0) {
            // 获取最后一位数字
            int num = n % 10;
            // 计算平方和
            sum += num*num;
            // 移除最后一位数字
            n = n/10; 
        }
        return sum;
    }
};

// 6.随机链表的复制
// 题目描述：
    // 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。 构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。返回复制链表的头节点。用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：val：一个表示 Node.val 的整数。random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。你的代码 只 接受原链表的头节点 head 作为传入参数。
// 解题思路：
    // 用一个哈希表存储原始链表和新链表之间的映射关系，遍历两次初始链表：
        // 第一次遍历，先把所有节点克隆出来；
        // 第二次遍历，把克隆结点的结构连接好；
    // 最后返回克隆之后的头节点
// 核心代码：
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        unordered_map<Node*, Node*> originToClone;

        // 第一次遍历：先把所有节点克隆出来
        for (Node* p = head; p != nullptr; p = p->next) {
            originToClone[p] = new Node(p->val);
        }

        // 第二次遍历：构造克隆结点的结构
        for (Node* p = head; p != nullptr; p = p->next) {
            // 注意点1：当前克隆节点应该指向下一个新的克隆结点，而不是原结点
            originToClone[p]->next = originToClone[p->next];
            originToClone[p]->random = originToClone[p->random];
        }

        return originToClone[head];
    }
};

// 7.两数之和
// 题目描述：
    // 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
// 解题思路：
    // 使用哈希表存储数组中元素和索引的映射关系，遍历数组，对于每个元素，判断 target - 当前元素是否在哈希表中；如果存在，则返回当前元素和 target - 当前元素的索引；
    // 注意：同一个元素不能使用两遍，所以需要在遍历数组时，跳过已经使用过的元素
// 核心代码：
class Solution {
public:    
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> map;
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            if (map.count(complement)) {
                return {map[complement], i};
            }
            map[nums[i]] = i;
        }
        return {};
    }
};