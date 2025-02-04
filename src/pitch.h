#ifndef PITCH_H
#define PITCH_H

#include <unistd.h>
#include <vector>
#include <cmath>
#include <deque>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <array>
#include <queue>
#include <algorithm>

#define NONE 0
#define ONE 1
#define TWO 2

using namespace std;

typedef int player_t;

template <typename T, size_t MaxSize>
class SmallDeque {
public:
    size_t size = 0;

    void push_front(const T& value) {
        // if (size < MaxSize) {
            front = (front - 1 + MaxSize) % MaxSize;
            buffer[front] = value;
            size++;
        // }
    }

    void push_back(const T& value) {
        // if (size < MaxSize) {
            buffer[back] = value;
            back = (back + 1) % MaxSize;
            size++;
        // }
    }

    void pop_front() {
        if (size > 0) {
            front = (front + 1) % MaxSize;
            size--;
        }
    }

    void pop_back() {
        if (size > 0) {
            back = (back - 1 + MaxSize) % MaxSize;
            size--;
        }
    }

    T& front_element() { return buffer[front]; }
    T& back_element() { return buffer[(back - 1 + MaxSize) % MaxSize]; }

private:
    std::array<T, MaxSize> buffer;
    size_t front = 0;
    size_t back = 0;

};

class Point {
public:
    int x;
    int y;
    explicit Point(int x = 0, int y = 0) : x(x), y(y) {}
};

class Edge {
public:
    Point a;
    Point b;
    int player;
    int x;
    int y;
    explicit Edge(Point a, Point b, int player = NONE) : a(a), b(b), player(player) {}
    bool operator==(const Edge &rhs) const {
        return a.x == rhs.a.x && a.y == rhs.a.y && b.x == rhs.b.x && b.y == rhs.b.y;
    }
    bool operator < (const Edge& other) const {
        int hash = x > y ? (x<<8) + y : (y<<8) + x;
        int hash2 = other.x > other.y ? (other.x<<8) + other.y : (other.y<<8) + other.x;
        return hash < hash2;
    }
};

class Path {
public:
    int a;
    int b;
    int hashCode;
    explicit Path(int a, int b) : a(a), b(b), hashCode((a>b)?((b<<16)+a):((a<<16)+b)) {}
    ~Path() { }
    Path& operator=(const Path& that) {
        a = that.a;
        b = that.b;
        hashCode = that.hashCode;

        return *this;
    }
    bool operator==(const Path &rhs) const {
        return hashCode == rhs.hashCode;
    }
    bool operator < (const Path& other) const {
        return hashCode < other.hashCode;
    }
    Path(const Path& that) : a(that.a), b(that.b), hashCode(that.hashCode) {}
};

static vector<Path> ALL_EDGES;
static vector<int> ALL_EDGES_INDEXES;
static vector<int> BALLS_MIRRORED;
static vector<int> EDGES_MIRRORED;

class Pitch {
public:
    int ball;
    int size,width,height;

    vector<uint16_t> matrix;
    vector<uint16_t> matrixNodes;

    vector<vector<int>> matrixNeighbours;

    explicit Pitch(int width, int height, bool halfLine = false) : width(width), height(height) {
        int w = width + 1;
        int h = height + 1;

        int wh = w * h;
        size = wh + 6;

        matrix.assign(size * size, 0);
        matrixNodes.assign(size, 0);

        // assign 2D points into matrix
        for (int i = 0; i < wh; i++) {
            int x = i / h;
            int y = i % h;
            matrix[i * size + i] = (x << 8) + y;
        }
        // goals
        matrix[wh * size + wh] = ((width / 2 - 1) << 8) + 0xFF;
        matrix[(wh + 1) * size + wh + 1] = ((width / 2) << 8) + 0xFF;
        matrix[(wh + 2) * size + wh + 2] = ((width / 2 + 1) << 8) + 0xFF;
        matrix[(wh + 3) * size + wh + 3] = ((width / 2 - 1) << 8) + h;
        matrix[(wh + 4) * size + wh + 4] = ((width / 2) << 8) + h;
        matrix[(wh + 5) * size + wh + 5] = ((width / 2 + 1) << 8) + h;

        //    matrix[(wh)*size+wh] = ((width/2)<<8) + 0xFF;
        //    matrix[(wh+1)*size+wh+1] = ((width/2+1)<<8) + h;

        // make neighbours
        for (int i = 0; i < size; i++) {
            makeNeighbours(i);
        }
        // remove some 'adjacency' from goals
        removeAdjacency(wh, h * (width / 2 - 2));
        removeAdjacency(wh + 2, h * (width / 2 + 2));
        removeAdjacency(wh + 3, h * (width / 2 - 1) - 1);
        removeAdjacency(wh + 5, h * (width / 2 + 3) - 1);

        // add edges except for goal
        for (int i = 0; i < wh - 1; i++) {
            for (int j = i + 1; j < wh; j++) {
                Point p = getPosition(i);
                Point p2 = getPosition(j);
                if ((p.x == 0 && p2.x == 0) && distance(p, p2) <= 1)
                    addEdge(i, j);
                else if ((p.x == width && p2.x == width) && distance(p, p2) <= 1)
                    addEdge(i, j);
                else if ((p.y == 0 && p2.y == 0) && distance(p, p2) <= 1) {
                    if (p.x < width / 2 - 1 || p.x >= width / 2 + 1) addEdge(i, j);
                } else if ((p.y == height && p2.y == height) && distance(p, p2) <= 1) {
                    if (p.x < width / 2 - 1 || p.x >= width / 2 + 1) addEdge(i, j);
                } else if (halfLine && (p.y == height / 2 && p2.y == height / 2) && distance(p, p2) <= 1)
                    addEdge(i, j);
            }
        }
        // and goals
        addEdge(wh, wh + 1);
        addEdge(wh + 1, wh + 2);
        addEdge(wh, h * (width / 2 - 1));
        addEdge(wh + 2, h * (width / 2 + 1));
        addEdge(wh + 3, wh + 4);
        addEdge(wh + 4, wh + 5);
        addEdge(wh + 3, h * (width / 2) - 1);
        addEdge(wh + 5, h * (width / 2 + 2) - 1);

        // create adjacency list
        matrixNeighbours.reserve(size);
        for (int i = 0; i < size; i++) {
            matrixNeighbours.push_back(getNeighbours(i));
            matrixNodes[i] += matrixNeighbours[i].size() << 4;
        }

        // create auxiliary arrays for goals
        goalArray.assign(size, NONE);
        almostGoalArray.assign(size, NONE);
        cutOffGoalArray.assign(size, NONE);
        for (int i = wh; i < size; i++) {
            goalArray[i] = (i - wh) / 3 == 0 ? ONE : TWO;
            almostGoalArray[i] = (i - wh) / 3 == 0 ? ONE : TWO;
            cutOffGoalArray[i] = (i - wh) / 3 == 0 ? ONE : TWO;
        }
        almostGoalArray[h * (width / 2 - 1)] = ONE;
        almostGoalArray[h * (width / 2 + 1)] = ONE;
        almostGoalArray[h * (width / 2) - 1] = TWO;
        almostGoalArray[h * (width / 2 + 2) - 1] = TWO;
        cutOffGoalArray[h * (width / 2 - 1)] = ONE;
        cutOffGoalArray[h * (width / 2)] = ONE;
        cutOffGoalArray[h * (width / 2 + 1)] = ONE;
        cutOffGoalArray[h * (width / 2) - 1] = TWO;
        cutOffGoalArray[h * (width / 2 + 1) - 1] = TWO;
        cutOffGoalArray[h * (width / 2 + 2) - 1] = TWO;

        matrixNodes[wh + 1]--;
        matrixNodes[wh + 1]--;
        matrixNodes[wh + 4]--;
        matrixNodes[wh + 4]--;

        // ball in center
        ball = width / 2 * h + h / 2;
    }

    void GENERATE_ALL_EDGES() {
        ALL_EDGES = getAllEdges();
        ALL_EDGES_INDEXES.resize(11025);
        for (int i=0; i < 316; i++) {
            ALL_EDGES_INDEXES[105 * ALL_EDGES[i].a + ALL_EDGES[i].b] = i;
            ALL_EDGES_INDEXES[105 * ALL_EDGES[i].b + ALL_EDGES[i].a] = i;
        }
        BALLS_MIRRORED.resize(105);
        for (int i=0; i < 105; i++) {
            BALLS_MIRRORED[i] = getMirroredIndex(i);
        }
        EDGES_MIRRORED.resize(316);
        for (int i=0; i < 316; i++) {
            EDGES_MIRRORED[i] = getMirroredPathIndex(i);
        }
    }

    void setPitch(const Pitch & pitch) {
        this->width = pitch.width;
        this->height = pitch.height;
        this->size = pitch.size;
        this->ball = pitch.ball;
        this->matrix = pitch.matrix;
        this->matrixNodes = pitch.matrixNodes;
        this->matrixNeighbours = pitch.matrixNeighbours;
    }

    void addEdge(int a, int b, int c = NONE) {
        matrix[a * size + b] |= 2;
        matrix[b * size + a] |= 2;
        if (c != NONE) {
            matrix[a * size + b] |= c == ONE ? 4 : 8;
            matrix[b * size + a] |= c == ONE ? 4 : 8;
        }
        matrixNodes[a]--;
        matrixNodes[b]--;
    }

    void removeEdge(int a, int b) {
        matrix[a * size + b] = 1;
        matrix[b * size + a] = 1;
        matrixNodes[a]++;
        matrixNodes[b]++;
    }

    void fillFreeNeighbours(vector<int> &ns, int index) {
        int n = matrixNodes[index] & 0x0F;
        ns.resize(n);

        for (int i = 0, j = 0; j < n; i++) {
            if (matrix[index * size + matrixNeighbours[index][i]] == 1) {
                ns[j] = matrixNeighbours[index][i];
                j++;
            }
        }
    }

    bool isGoalReachable(player_t player) {
        vector<uint8_t> visited(size,0);
        vector<int> stos; stos.reserve(10);
        stos.push_back(ball);
        visited[ball] = true;

        while (!stos.empty()) {
            int q = stos.back();
            stos.pop_back();
            int t = q * size;
            for (auto &v : matrixNeighbours[q]) {
                if (visited[v] || matrix[t + v] > 1) continue;
                if (almostGoalArray[v] == player) {
                    return true;
                } else {
                    int n = matrixNodes[v] & 0x0F;
                    if (n > 1 && n<matrixNodes[v]>> 4) {
                        stos.push_back(v);
                    }
                }
                visited[v] = true;
            }
            visited[q] = true;
        }
        return false;
    }

    player_t goal(int index) {
        return goalArray[index];
    }

    int getNeighbour(int dx, int dy) {
        Point point = getPosition(ball);
        for (auto &n : matrixNeighbours[ball]) {
            if (matrix[ball * size + n] > 1) continue;
            Point p = getPosition(n);
            if (point.x + dx == p.x && point.y + dy == p.y) return n;
        }
        return -1;
    }

    int getNeighbour(int index, int dx, int dy) {
        Point point = getPosition(index);
        for (auto &n : matrixNeighbours[index]) {
            // if (matrix[ball * size + n] > 1) continue;
            Point p = getPosition(n);
            if (point.x + dx == p.x && point.y + dy == p.y) return n;
        }
        return -1;
    }

    bool isBlocked(int index) {
        return (matrixNodes[index] & 0x0F) == 0;
    }

    bool isAlmostBlocked(int index) {
        return (matrixNodes[index] & 0x0F) <= 1;
    }

    bool passNext(int index) {
        return (matrixNodes[index] & 0x0F)<(matrixNodes[index] >> 4);
    }

    bool passNextDone(int index) {
        return (matrixNodes[index] & 0x0F) < (matrixNodes[index] >> 4) - 1;
    }

    bool passNextDone2(int index) {
        return (matrixNodes[index] & 0x0F) < (matrixNodes[index] >> 4) - 2;
    }

    string shortWinningMoveForPlayer(player_t player) {
        player = player==ONE?TWO:ONE;
        queue<int> kolejka;
        vector<int> parents(size,-1);
        vector<int> distances(size,99);
        kolejka.push(ball);
        distances[ball] = 0;
        int e = -1;

        while (!kolejka.empty()) {
            int q = kolejka.front();
            kolejka.pop();
            for (auto &v : matrixNeighbours[q]) {
                if (matrix[q*size+v] > 1) continue;
                player_t g = goalArray[v];
                if (g == NONE && passNext(v) && distances[q]+1 < distances[v]) {
                    parents[v] = q;
                    distances[v] = distances[q]+1;
                    kolejka.push(v);
                } else if (g == player && (e == -1 || distances[q]+1 < distances[e])) {
                    e = v;
                    parents[e] = q;
                    distances[e] = distances[q]+1;
                }
            }
        }

        string move = "";
        while (parents[e] != -1) {
            move += getDistanceChar(parents[e],e);
            e = parents[e];
        }
        reverse(move.begin(),move.end());

        return move;
    }

    bool onlyOneEmpty() {
        vector<bool> visited(size,false);
        vector<int> stos;

        stos.push_back(ball);
        visited[ball] = true;

        while (!stos.empty()) {
            int q = stos.back();
            stos.pop_back();
            int t = q * size;
            for (auto &v : matrixNeighbours[q]) {
                if (visited[v] || matrix[t + v] > 1) continue;
                if (!isAlmostBlocked(v)) {
                    if (passNext(v)) {
                        stos.push_back(v);
                    } else {
                        stos.resize(0);
                        return false;
                    }
                } else if (goalArray[v] != NONE) {
                    stos.resize(0);
                    return false;
                }
                visited[v] = true;
            }
        }

        return true;
    }

    bool onlyTwoEmpty() {
        vector<bool> visited(size,false);
        vector<int> stos;

        stos.push_back(ball);
        visited[ball] = true;

        int c = 0;

        while (!stos.empty()) {
            int q = stos.back();
            stos.pop_back();
            int t = q * size;
            for (auto &v : matrixNeighbours[q]) {
                if (visited[v] || matrix[t + v] > 1) continue;
                if (!isAlmostBlocked(v)) {
                    if (passNext(v)) {
                        stos.push_back(v);
                    } else {
                        c++;
                        if (c > 1) {
                            stos.resize(0);
                            return false;
                        } else {
                            stos.push_back(v);
                        }
                    }
                } else if (goalArray[v] != NONE) {
                    stos.resize(0);
                    return false;
                }
                visited[v] = true;
            }
        }

        return true;
    }

    vector<int> fillBlockedKonce() {
        vector<bool> visited(size,false);
        vector<int> stos;

        vector<int> ns(8);
        vector<int> edges;

        stos.push_back(ball);
        vector<int> rodzice(size, -1);
        vector<int> konce;
        while (!stos.empty()) {
            int q = stos.back();
            stos.pop_back();
            int t = q * size;
            for (auto &v : matrixNeighbours[q]) {
                if (visited[v] || matrix[t + v] > 1) continue;
                if (goalArray[v] != NONE) {
                    // konce.push_back(v);
                } else {
                    if (passNext(v) && !isAlmostBlocked(v)) {
                        rodzice[v] = q;
                        stos.push_back(v);
                    } else {
                        if (isAlmostBlocked(v)) {
                            rodzice[v] = q;
                            konce.push_back(v);
                        }
                    }
                }
                visited[v] = true;
            }
            visited[q] = true;
        }

        for (auto &k : konce) {
            addEdge(k, rodzice[k]);
            edges.push_back(k);
            edges.push_back(rodzice[k]);
            int t = rodzice[k];
            while ((matrixNodes[t] & 0x0F) == 1 && rodzice[t] != -1) {
                addEdge(t, rodzice[t]);
                edges.push_back(t);
                edges.push_back(rodzice[t]);
                t = rodzice[t];
            }
        }

        return edges;
    }


    bool shouldCheckForGameOver(player_t player) {
        vector<bool> visited(size,false);
        vector<int> distances(size, 32768);

        deque<int> kolejka;
        int s = player == ONE ? (size - 5) : (size - 2);
        kolejka.push_back(s);
        distances[s] = 1;

        int kroki = 0;

        while (!kolejka.empty()) {
            kroki++;
            int q = kolejka.front();
            kolejka.pop_front();
            if (distances[q] > 2) return false;
            bool pq = passNext(q);
            for (auto &v : matrixNeighbours[q]) {
                if (matrix[q * size + v] > 1) continue;
                if (visited[v]) continue;
                if (pq) {
                    if (distances[v] > distances[q]) {
                        distances[v] = distances[q];
                        if (goalArray[v] == NONE) {
                            if (passNext(v))
                                kolejka.push_front(v);
                            else
                                kolejka.push_back(v);
                        }
                    }
                } else if (distances[v] > distances[q] + 1) {
                    distances[v] = distances[q] + 1;
                    if (goalArray[v] == NONE) {
                        if (passNext(v))
                            kolejka.push_front(v);
                        else
                            kolejka.push_back(v);
                    }
                }

                if (v == ball) return true;
                visited[v] = true;
            }
            visited[q] = true;
        }

        return true;
    }

    bool isNextMoveGameover(player_t player) {
        vector<int> stos;
        vector<short> visited(size);
        stos.push_back(ball);
        visited[ball] = true;

        while (!stos.empty()) {
            int q = stos.back();
            stos.pop_back();
            int t = q*size;
            for (auto &v : matrixNeighbours[q]) {
                if (visited[v] || matrix[t+v] > 1) continue;
                if (almostGoalArray[v] == player) {
                    stos.resize(0);
                    return true;
                } else {
                    int n = matrixNodes[v]&0x0F;
                    if (n > 1 && n < matrixNodes[v]>>4) {
                        stos.push_back(v);
                    }
                }
                visited[v] = true;
            }
            visited[q] = true;
        }
        return false;
    }

    bool isCutOffFromOpponentGoal(player_t player) {
        player = player == ONE ? TWO : ONE;
        vector<uint8_t> visited(size,0);
        vector<int> stos; stos.reserve(10);
        stos.push_back(ball);
        visited[ball] = true;

        while (!stos.empty()) {
            int q = stos.back();
            stos.pop_back();
            int t = q * size;
            for (auto &v : matrixNeighbours[q]) {
                if (visited[v] || matrix[t + v] > 1) continue;
                if (cutOffGoalArray[v] == player) {
                    return false;
                } else if ((matrixNodes[v] & 0x0F) > 1) {
                    stos.push_back(v);
                }
                visited[v] = true;
            }
            visited[q] = true;
        }

        return true;
    }

    char getDistanceChar(int a, int b) {
        Point p1 = getPosition(a);
        Point p2 = getPosition(b);
        int dx = p2.x - p1.x;
        int dy = p2.y - p1.y;
        if (dx == 0) {
            if (dy == -1) return '0';
            return '4';
        } else if (dx == 1) {
            if (dy == -1) return '1';
            if (dy == 0) return '2';
            return '3';
        } else {
            if (dy == -1) return '7';
            if (dy == 0) return '6';
            if (dy == 1) return '5';
        }
        return -1;
    }

    Point getPosition(int index) {
        int x = (matrix[index * size + index] >> 8) & 0xFF;
        int y = matrix[index * size + index] & 0xFF;
        if (y == 0xFF) y = -1;
        return Point(x, y);
    }

    int getMirroredIndex(int index) {
        Point p = getPosition(index);
        for (int i=0; i < size; i++) {
            Point p2 = getPosition(i);
            if (p.y == height-p2.y && p.x == width-p2.x) {
                return i;
            }
        }
        return -1;
    }

    int getMirroredPathIndex(int index) {
        auto & path = ALL_EDGES[index];
        for (int i=0; i < 105; i++) {
            for (int j=0; j < 105; j++) {
                if (i == j) continue;
                int index2 = ALL_EDGES_INDEXES[105*i + j];
                auto & path2 = ALL_EDGES[index2];
                if (path.a == getMirroredIndex(path2.a) && path.b == getMirroredIndex(path2.b)) {
                    return index2;
                }
                if (path.b == getMirroredIndex(path2.a) && path.a == getMirroredIndex(path2.b)) {
                    return index2;
                }
            }
        }
        cout << "x" << endl;
        return -1;
    }

    int getDistancesToGoal(player_t player) {
        vector<int> distances = vector<int>(size, height);
        vector<bool> visited(size,false);
        vector<int> stos;

        deque<int> kolejka;
        int s = (player == ONE) ? (size - 5) : (size - 2);
        kolejka.push_back(s);
        distances[s] = 1;

        while (!kolejka.empty()) {
            int q = kolejka.front();
            kolejka.pop_front();
            bool pq = passNext(q);
            for (auto &v : matrixNeighbours[q]) {
                if (matrix[q * size + v] > 1) continue;
                if (visited[v]) continue;
                if (pq) {
                    if (distances[v] > distances[q]) {
                        distances[v] = distances[q];
                        if (passNext(v))
                            kolejka.push_front(v);
                        else
                            kolejka.push_back(v);
                    }
                } else if (distances[v] > distances[q] + 1) {
                    distances[v] = distances[q] + 1;
                    if (passNext(v))
                        kolejka.push_front(v);
                    else
                        kolejka.push_back(v);
                }

                if (v == ball) return distances[ball];
                visited[v] = true;
            }
            visited[q] = true;
        }

        return height;
    }

    void calculateDistances(int source, vector<int> & distances) {
        vector<bool> visited(size,false);

        deque<int> kolejka;
        kolejka.push_back(source);
        distances[source] = 1;

        while (!kolejka.empty()) {
            int q = kolejka.front();
            kolejka.pop_front();
            bool pq = passNext(q);
            for (auto &v : matrixNeighbours[q]) {
                if (matrix[q * size + v] > 1) continue;
                if (visited[v]) continue;
                if (pq) {
                    if (distances[v] > distances[q]) {
                        distances[v] = distances[q];
                        if (passNext(v))
                            kolejka.push_front(v);
                        else
                            kolejka.push_back(v);
                    }
                } else if (distances[v] > distances[q] + 1) {
                    distances[v] = distances[q] + 1;
                    if (passNext(v))
                        kolejka.push_front(v);
                    else
                        kolejka.push_back(v);
                }

                visited[v] = true;
            }
            visited[q] = true;
        }
    }

    void calculateDistances2(int source, vector<int> & distances) {
        vector<uint8_t> visited(size,0);
        SmallDeque<int, 64> kolejka;
        kolejka.push_back(source);
        distances[source] = 1;

        while (kolejka.size != 0) {
            int q = kolejka.front_element();
            kolejka.pop_front();
            bool pq = passNext(q);
            for (auto &v : matrixNeighbours[q]) {
                if (matrix[q * size + v] > 1) continue;
                if (visited[v]) continue;
                if (pq) {
                    if (distances[v] > distances[q]) {
                        distances[v] = distances[q];
                        if (passNext(v))
                            kolejka.push_front(v);
                        else
                            kolejka.push_back(v);
                    }
                } else if (distances[v] > distances[q] + 1) {
                    distances[v] = distances[q] + 1;
                    if (passNext(v))
                        kolejka.push_front(v);
                    else
                        kolejka.push_back(v);
                }

                visited[v] = true;
            }
            visited[q] = true;
        }
    }

    uint64_t getHash() {
        uint64_t hash = 0L;
        for (int i = 1; i < size; i++) {
            for (int j = 0; j < i; j++) {
                if ((matrix[i * size + j] & 2) > 0) {
                    uint64_t p = ((i+1) << 16) + (j+1);
                    hash ^= murmurHash3(202289 * p);
                }
            }
        }
        hash ^= murmurHash3(101 * ball + 1);
        return hash;
    }

    uint64_t murmurHash3(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53L;
        x ^= x >> 33;
        return x;
    }

    vector<int> getTuplesEdgesPov() {
        vector<int> edges; edges.reserve(316);
        for (int i=0; i < 316; i++) {
            if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                edges.push_back(i);
            }
        }
        edges.push_back(316+ball);
        return edges;
    }

    vector<int> getTuplesEdges() {
        vector<int> edges(415);
        for (int i=0; i < 316; i++) {
            edges[i] = existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b) ? 1 : 0;
        }
        edges[316+ball] = 1;
        return edges;
    }

    int getSquare(int i) {
        int index = 0;
        if (existsEdge(i,i+1)) index += 1;
        if (existsEdge(i,i+9)) index += 2;
        if (existsEdge(i,i+10)) index += 4;
        if (existsEdge(i+1,i+9)) index += 8;
        if (existsEdge(i+1,i+10)) index += 16;
        if (existsEdge(i+9,i+10)) index += 32;
        if (ball == i) index += 64;
        if (ball == i+1) index += 128;
        if (ball == i+9) index += 256;
        if (ball == i+10) index += 512;
        return index;
    }

    vector<int> squares = { 0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52 };

    vector<int> getTuplesEdgesBase(player_t player) {
        vector<int> edges; edges.reserve(316);
        if (player == ONE) {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(i);
                }
            }
        } else {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(EDGES_MIRRORED[i]);
                }
            }
        }
        return edges;
    }

    vector<int> getTuplesEdges(player_t player) {
        vector<int> edges; edges.reserve(316);
        if (player == ONE) {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(i);
                }
            }
            edges.push_back(316+ball);
        } else {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(EDGES_MIRRORED[i]);
                }
            }
            edges.push_back(316+BALLS_MIRRORED[ball]);
        }
        return edges;
    }

    vector<int> getTuplesEdges2(player_t player) {
        // int c = player == ONE ? 0 : 49152;
        // vector<int> indexes; indexes.reserve(100);
        // for (size_t i=0; i < squares.size(); i++) {
        //     indexes.push_back(c+1024*i+getSquare(i));
        // }
        // return indexes;

        vector<int> edges; edges.reserve(316);
        vector<int> distances(105,9);
        calculateDistances(ball,distances);
        int n = player == ONE ? 0 : 435;
        for (int i=0; i < 316; i++) {
            if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                edges.push_back(n+i);
            }
        }
        edges.push_back(316+n+distances[100]);
        edges.push_back(316+n+10+distances[103]);
        // for (int i=0; i < 105; i++) {
        //     edges.push_back(316+n+10*i+distances[i]);
        // }
        //edges.push_back(n+192+ball);
        edges.push_back(n+336+ball);
        return edges;

        // vector<int> edges; edges.reserve(316);
        // int n = player == ONE ? 0 : 415;
        // for (int i=0; i < 316; i++) {
        //     if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
        //         edges.push_back(n+i);
        //     }
        // }
        // edges.push_back(n+316+ball);
        // return edges;


        // int n = player == ONE ? 0 : 415;
        // vector<int> edges(830);
        // for (int i=0; i < 316; i++) {
        //     edges[n+i] = existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b) ? 1 : 0;
        // }
        // edges[n+316+ball] = 1;
        // return edges;
    }

    vector<int> getTuplesEdgesBase2(player_t player) {
        vector<int> edges; edges.reserve(316);
        int n = player == ONE ? 0 : 435;
        for (int i=0; i < 316; i++) {
            if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                edges.push_back(n+i);
            }
        }
        return edges;
    }

    vector<int> getTuplesEdgesBase3(player_t player) {
        vector<int> edges; edges.reserve(316);
        if (player == ONE) {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(i);
                }
            }
        } else {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(EDGES_MIRRORED[i]);
                }
            }
        }
        return edges;
    }

    vector<int> getTuplesEdges3(player_t player) {
        vector<int> edges; edges.reserve(316);
        vector<int> distances(105,9);
        calculateDistances(ball,distances);
        if (player == ONE) {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(i);
                }
            }
            edges.push_back(316+distances[100]);
            edges.push_back(316+10+distances[103]);
            edges.push_back(336+ball);
        } else {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(EDGES_MIRRORED[i]);
                }
            }
            edges.push_back(316+distances[103]);
            edges.push_back(316+10+distances[100]);
            edges.push_back(336+BALLS_MIRRORED[ball]);
        }
        return edges;
    }

    vector<int> getTuplesEdgesBase4(player_t player) {
        vector<int> edges; edges.reserve(316);
        if (player == ONE) {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(i);
                }
            }
        } else {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(EDGES_MIRRORED[i]);
                }
            }
        }
        return edges;
    }

    vector<int> getTuplesEdges4(player_t player) {
        vector<int> edges; edges.reserve(316);
        vector<int> distances(105,9);
        calculateDistances(ball,distances);
        if (player == ONE) {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(i);
                }
            }
            for (int i=0; i < 105; i++) {
                edges.push_back(316+10*i+distances[i]);
            }
            edges.push_back(1366+ball);
        } else {
            for (int i=0; i < 316; i++) {
                if (existsEdge(ALL_EDGES[i].a,ALL_EDGES[i].b)) {
                    edges.push_back(EDGES_MIRRORED[i]);
                }
            }
            for (int i=0; i < 105; i++) {
                edges.push_back(316+10*i+distances[BALLS_MIRRORED[i]]);
            }
            edges.push_back(1366+BALLS_MIRRORED[ball]);
        }
        return edges;
    }

    vector<Path> getAllEdges() {
        vector<Path> edges; edges.reserve(316);
        for (int i=1;i<size;i++) {
            for (int j=0;j<i;j++) {
                if ((matrix[i*size+j]) > 0 && (matrix[i*size+j]) != 3) {
                    edges.push_back(Path(i,j));
                }
            }
        }
        return edges;
    }

    vector<Edge> getEdges() {
        vector<Edge> edges;
        for (int i=1;i<size;i++) {
            for (int j=0;j<i;j++) {
                if ((matrix[i*size+j]&2) > 0) {
                    Point a = getPosition(i);
                    Point b = getPosition(j);
                    int p = matrix[i*size+j];
                    Edge edge(a,b);
                    edge.x = i;
                    edge.y = j;
                    if ((matrix[i*size+j]&4) > 0) {
                        edge.player = ONE;
                    } else if ((matrix[i*size+j]&8) > 0) {
                        edge.player = TWO;
                    }
                    edges.push_back(edge);
                }
            }
        }
        return edges;
    }

    vector<Edge> getEdgesMirrored() {
        vector<Edge> edges;
        for (int i=1;i<size;i++) {
            for (int j=0;j<i;j++) {
                if ((matrix[i*size+j]&2) > 0) {
                    Point a = getPosition(getMirroredIndex(i));
                    Point b = getPosition(getMirroredIndex(j));
                    int p = matrix[i*size+j];
                    Edge edge(a,b);
                    edge.x = i;
                    edge.y = j;
                    if ((matrix[i*size+j]&4) > 0) {
                        edge.player = ONE;
                    } else if ((matrix[i*size+j]&8) > 0) {
                        edge.player = TWO;
                    }
                    edges.push_back(edge);
                }
            }
        }
        return edges;
    }

    bool existsEdge(int a, int b) {
        return (matrix[a*size+b] & 2) != 0;
    }

    int nextNode(int index, char c) {
        switch(c) {
        case '5': return getNeighbour(index,-1,1);
        case '4': return getNeighbour(index,0,1);
        case '3': return getNeighbour(index,1,1);
        case '6': return getNeighbour(index,-1,0);
        case '2': return getNeighbour(index,1,0);
        case '7': return getNeighbour(index,-1,-1);
        case '0': return getNeighbour(index,0,-1);
        case '1': return getNeighbour(index,1,-1);
        }
        return -1;
    }

    //private:
    vector<player_t> goalArray;
    vector<player_t> almostGoalArray;
    vector<player_t> cutOffGoalArray;

    int distance(Point p1, Point p2) {
        return (int)sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    }

    vector<int> getNeighbours(int index) {
        vector<int> neibghours;
        neibghours.reserve(8);
        for (int i = 0; i < size; i++) {
            if (i == index) continue;
            if ((matrix[index * size + i] & 1) == 1) neibghours.push_back(i);
        }
        return neibghours;
    }

    void removeAdjacency(int a, int b) {
        matrix[a * size + b] = 0;
        matrix[b * size + a] = 0;
        matrixNodes[a]--;
        matrixNodes[b]--;
    }

    void makeNeighbours(int index) {
        Point point = getPosition(index);

        for (int i = 0; i < size; i++) {
            if (i == index) continue;
            Point p = getPosition(i);
            if (distance(point, p) <= 1) {
                matrixNodes[i]++;
                matrix[index * size + i] = 1;
                matrix[i * size + index] = 1;
            }
        }
    }
};

#endif // PITCH_H
