#include <bits/stdc++.h>
using namespace std;

// 1. typedf和define的区别
typedef unsigned int uint; // 为已有类型创建一个新的类型别名。由编译器处理，在语法树中保留类型信息。
uint a = 10; // 相当于：unsigned int a = 10;

#define PI 3.14 // 定义宏，文本替换。由预处理器在编译前进行“纯文本”替换。
#define uint unsigned int // 无类型检查（文本替换，易出错）。

// 2.define和inline的区别
#define SQUARE(x) ((x) * (x))
int a = SQUARE(5);  // 预处理器完成的字符串替换，不进行类型检查或语义分析。无调试信息，不便于调试。没有作用域概念，容易污染命名空间。

// 请求编译器将函数“展开”为函数体代码，减少函数调用开销。
inline int square(int x) {
    return x * x;
}
// 编译器级别优化，支持类型检查，有作用域。
int a = square(5);  // 编译器可能内联为：int a = 5 * 5;




