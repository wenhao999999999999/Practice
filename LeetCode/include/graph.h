// 图结构
#include <bits/stdc++.h>
using namespace std;

// 邻接表实现的图结构
class Graph {
public:
    // 存储相邻节点及边的权重
    struct Edge {
        int to;
        int weight;

        // 构造函数，应该写在结构体内部
        Edge(int to, int weight) {
            this->to = to;
            this->weight = weight;
        }
    };
private:
    // 邻接表
    vector<vector<Edge>> graph;

public: 
    // 构造函数，初始化图的节点数
    Graph(int n) {
        graph.resize(n);
    }

    // 添加边
    void addEdge(int from, int to, int weight) {
        graph[from].push_back(Edge(to, weight));
    }

    // 删除边
    void removeEdge(int from, int to) {
        for (auto it = graph[from].begin(); it!= graph[from].end(); it++) {
            if (it->to == to) {
                graph[from].erase(it);
                break;
            }
        }
    }

    // 查，返回一条边的权重
    int getEdgeWeight(int from, int to) {
        for (auto it = graph[from].begin(); it!= graph[from].end(); it++) {
            if (it->to == to) {
                return it->weight;
            }
        return -1; // 没有找到边
        }
    }

    // 查，返回某个节点的所有邻居节点
    const vector<Edge>& getNeighbors(int from) {
        return graph[from];
    }

    virtual int size() = 0;
};

// 邻接矩阵实现的图结构
class Graph {
public:
    // 存储相邻节点及边的权重
    struct Edge {
        int to;
        int weight;

        // 构造函数，应该写在结构体内部
        Edge(int to, int weight) {
            this->to = to;
            this->weight = weight;
        }
    };
private:
    // 邻接矩阵
    vector<vector<int>> graph;

public: 
    // 构造函数，初始化图的节点数
    Graph(int n) {
        graph.resize(n, vector<int>(n, 0));
    }

    // 添加边
    void addEdge(int from, int to, int weight) {
        graph[from][to] = weight;
    }

    // 删除边
    void removeEdge(int from, int to) {
        graph[from][to] = 0;
    }

    // 查，判断两个节点是否相邻
    bool isAdjacent(int from, int to) {
        return graph[from][to] != 0;
    }

    // 查，返回一条边的权重
    int getEdgeWeight(int from, int to) {
        return graph[from][to];
    }

    // 查，返回某个节点的所有邻居节点
    vector<Edge> getNeighbors(int from) {
        vector<Edge> neighbors;
        for (int i = 0; i < graph[from].size(); i++) {
            if (graph[from][i] != 0) {
                neighbors.push_back(Edge(i, graph[from][i]));
            }
        }
        return neighbors;
    }
};

// Union Find 并查集原理
// 并查集（Union Find）结构是 二叉树结构 的衍生，用于高效解决无向图的连通性问题，可以在 O(1) 时间内合并两个连通分量，在 O(1) 时间内查询两个节点是否连通，在 O(1) 时间内查询连通分量的数量。

// 并查集（Union Find）结构提供如下 API：
class UF {
public:
    // 初始化并查集，包含 n 个节点，时间复杂度 O(n)
     UF(int n);

    // 连接节点 p 和节点 q，时间复杂度 O(1)
    void unions(int p, int q);

    // 查询节点 p 和节点 q 是否连通（是否在同一个连通分量内），时间复杂度 O(1)
    bool connected(int p, int q);

    // 查询当前的连通分量数量，时间复杂度 O(1)
    int count();
};

// 并查集本质上还是树结构的延伸。如果我们想办法把同一个连通分量的节点都放到同一棵树中，把这棵树的根节点作为这个连通分量的代表，那么我们就可以高效实现上面的操作了。并查集底层其实是一片森林（若干棵多叉树），每棵树代表一个连通分量：
// connected(p, q)：只需要判断 p 和 q 所在的多叉树的根节点，若相同，则 p 和 q 在同一棵树中，即连通，否则不连通。
// count()：只需要统计一下总共有多少棵树，即可得到连通分量的数量。
// union(p, q)：只需要将 p 节点所在的这棵树的根节点，接入到 q 节点所在的这棵树的根节点下面，即可完成连接操作。注意这里并不是 p, q 两个节点的合并，而是两棵树根节点的合并。因为 p, q 一旦连通，那么他们所属的连通分量就合并成了同一个更大的连通分量。
// 综上，并查集中每个节点其实不在乎自己的子节点是谁，只在乎自己的根节点是谁，所以一个并查集节点类似于下面这样：
// class UFNode {
//     // 节点 id 编号
//     int id;

//     // 指向父节点的指针
//     // 根节点的 parent 指针为空
//     UFNode parents;
// };

// 所以并查集算法最终的目标，就是要尽可能降低树的高度，如果能保持树高为常数，那么上述方法的复杂度就都是 O(1) 了。

// 图结构的 DFS/BFS 遍历
// 图的遍历比多叉树的遍历多了一个 visited 数组，用来记录被遍历过的节点，避免遇到环时陷入死循环。
// 深度优先搜索（DFS）
// (1) 遍历图的所有节点（一维 Visited 数组）
// void traverse(const Graph& graph, int s, std::vector<bool>& visited) {
//     // base case
//     if (s < 0 || s >= graph.size()) {
//         return;
//     }
//     if (visited[s]) {
//         // 防止死循环
//         return;
//     }
//     // 前序位置
//     visited[s] = true;
//     std::cout << "visit " << s << std::endl;
//     for (const Graph::Edge& e : graph.neighbors(s)) {
//         traverse(graph, e.to, visited);
//     }
//     // 后序位置
// }

// (2) 遍历所有变（二维 Visited 数组）
// 图节点
class Vertex {
public:
    int id;
    vector<Vertex*> neighbors;
    
    Vertex(int i = 0) : id(i) {}
};

// 遍历图的边
// 需要一个二维 visited 数组记录被遍历过的边，visited[u][v] 表示边 u->v 已经被遍历过
void traverseEdges(Vertex* s, vector<vector<bool>>& visited) {
    // base case
    if (s == nullptr) {
        return;
    }
    for (Vertex* neighbor : s->neighbors) {
        // 如果边已经被遍历过，则跳过
        if (visited[s->id][neighbor->id]) {
            continue;
        }
        // 标记并访问边
        visited[s->id][neighbor->id] = true;
        cout << "visit edge: " << s->id << " -> " << neighbor->id << endl;
        traverseEdges(neighbor, visited);
    }
}

// (3) 遍历所有路径(onpath 数组)
// 对于图结构来说，由起点 src 到目标节点 dest 的路径可能不止一条。我们需要一个 onPath 数组，在进入节点时（前序位置）标记为正在访问，退出节点时（后序位置）撤销标记，这样才能遍历图中的所有路径，从而找到 src 到 dest 的所有路径：
// 下面的算法代码可以遍历图的所有路径，寻找从 src 到 dest 的所有路径
// onPath 和 path 记录当前递归路径上的节点
// vector<bool> onPath(graph.size());
// list<int> path;

// void traverse(Graph& graph, int src, int dest) {
//     // base case
//     if (src < 0 || src >= graph.size()) {
//         return;
//     }
//     if (onPath[src]) {
//         // 防止死循环（成环）
//         return;
//     }
//     if (src == dest) {
//         // 找到目标节点
//         cout << "find path: ";
//         for (auto it = path.begin(); it != path.end(); it++) {
//             cout << *it << "->";
//         }
//         cout << dest << endl;
//         return;
//     }

//     // 前序位置
//     onPath[src] = true;
//     path.push_back(src);
//     for (const Edge& e : graph.neighbors(src)) {
//         traverse(graph, e.to, dest);
//     }
//     // 后序位置
//     path.pop_back();
//     onPath[src] = false;
// }

// 广度优先搜索（BFS）
    // 图结构的广度优先搜索其实就是多叉树的层序遍历，无非就是加了一个 visited 数组来避免重复遍历节点。
    // BFS 遍历也需要区分遍历所有「节点」和遍历所有「路径」，但是实际上 BFS 算法一般只用来寻找那条最短路径

// 从 s 开始 BFS 遍历图的所有节点，且记录遍历的步数
// void bfs (const Graph& graph, int s) {
//     //  vector<bool> visited(graph.size(), false);
//     queue<int> q;
//     q.push(s);
//     // visited[s] = true;

//     // 记录从 s 开始走到当前节点的步数
//     int step = 0;

//     while(!q.empty()) {
//         int size = q.size();
//         int curNode = q.front();
//         q.pop();

//         for (const Edge& e : graph.neighbors(cur)) {
//             if (visited[e.to]) {
//                 continue;
//             }

//             q.push(e.to);
//             visited[e.to] = true;
//         }
//         step++;
//     }

    
// }
        