#include <iostream>
using namespace std;

class Animal {
public:
    // 构造函数
    Animal() {cout << "Animal 构造函数" << endl; }

    // 析构函数
    virtual ~Animal() {cout << "Animal 析构函数" << endl; }

    // 虚函数,这个函数不会修改类的成员变量
    virtual void speak() const {
        cout << "Animal 在说话" << endl; 
    }
};

// 派生类 Dog
class Dog : public Animal {
public:
    Dog() {cout << "Dog 构造函数" << endl; }
    ~Dog() {cout << "Dog 虚构函数" << endl; }

    //明确表示这个函数是要重写基类的虚函数
    void speak() const override {
        cout << "Dog: 汪汪汪" << endl;
    }

};

// 派生类 Cat
class Cat : public Animal {
public:
    Cat() { cout << "Cat 构造函数" << endl; }
    ~Cat() { cout << "Cat 析构函数" << endl; }

    void speak() const override {
        cout << "Cat: 喵喵喵！" << endl;
    }
};

int main() {
    // 用基类指针指向派生类对象
    // 用统一的基类接口（Animal）来管理多个派生类对象
    Animal* a1 = new Dog(); // 为了实现 运行时多态（Dynamic Polymorphism）
    Animal* a2 = new Cat(); // 好处是：支持统一处理

    a1->speak();
    a2->speak();

    delete a1;
    delete a2;


    return 0;
}