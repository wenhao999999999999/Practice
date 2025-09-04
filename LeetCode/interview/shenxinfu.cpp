// #include <bits/stdc++.h>
// using namespace std;


// 已知散点坐标，求包围所有散点的矩形的最小面积
// struct Point{
// float x;
// float y;
// };
// std::vector<Point> points;

// class Solution {
// public:
//     float MinArea(const std::vector<Point> &points, float &x_centre, float &y_centre, float &angle, float &x_size, float &y_size){

//     float area;
//     return area;
//     }
// };


#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>

constexpr double PI = 3.141592653589793;

struct Point {
    double x, y;
};

// 将点绕原点逆时针旋转 angle 弧度
Point rotate(const Point& p, double angle) {
    double cos_a = cos(angle);
    double sin_a = sin(angle);
    return {
        p.x * cos_a - p.y * sin_a,
        p.x * sin_a + p.y * cos_a
    };
}

// 计算面积函数 A(theta) = width * height
double compute_area_at_angle(const std::vector<Point>& points, double theta, double& w_out, double& h_out) {
    double x_min = 1e9, x_max = -1e9;
    double y_min = 1e9, y_max = -1e9;

    for (const auto& p : points) {
        Point rp = rotate(p, -theta); // 坐标系旋转
        x_min = std::min(x_min, rp.x);
        x_max = std::max(x_max, rp.x);
        y_min = std::min(y_min, rp.y);
        y_max = std::max(y_max, rp.y);
    }

    w_out = x_max - x_min;
    h_out = y_max - y_min;
    return w_out * h_out;
}

int main() {
    std::vector<Point> points = {
        {1, 1}, {2, 2}, {3, 1}, {2.5, 3}, {1.5, 3.5},
        {4, 2.8}, {2.8, 1.2}, {1.2, 1.5}
    };

    double best_theta_deg = 0;
    double min_area = std::numeric_limits<double>::max();
    double best_w = 0, best_h = 0;

    for (double deg = 0; deg < 180; deg += 1.0) {
        double theta = deg * PI / 180.0;
        double w, h;
        double area = compute_area_at_angle(points, theta, w, h);
        if (area < min_area) {
            min_area = area;
            best_theta_deg = deg;
            best_w = w;
            best_h = h;
        }
    }

    std::cout << "最小面积: " << min_area << std::endl;
    std::cout << "最优角度: " << best_theta_deg << "°" << std::endl;
    std::cout << "矩形宽: " << best_w << ", 高: " << best_h << std::endl;

    return 0;
}

