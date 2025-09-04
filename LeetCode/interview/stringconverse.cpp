#include <bits/stdc++.h>
using namespace std;

//CVTE实习手撕代码20250609
class Solution {
public:
    string reverseString (string& s) {
        if(s.empty() || s.size() == 1) return s;
        int left = 0;
        int right = s.size() - 1;
        while(left <= right) {
            char c = s[left];
            s[left] = s[right];
            s[right] = c;
            left++;
            right--; 
        }
        return s;
    }
};

int main() {
    Solution sol;
    string s1 = "123";
    cout << sol.reverseString(s1) << endl;
    return 0;
} 