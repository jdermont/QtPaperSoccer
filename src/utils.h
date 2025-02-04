#ifndef UTILS_H
#define UTILS_H

#include <deque>
#include <vector>
#include <cstdint>

using namespace std;

class LimitedCapacityQueue {
public:
    size_t size;

    explicit LimitedCapacityQueue(size_t size) : size(size) {

    }

    void add(pair<vector<int16_t>,float> element) {
        if (deq.size() >= size) {
            deq.pop_front();
        }
        deq.push_back(element);
    }

    void clear() {
        deq.clear();
    }

    int getSize() {
        return deq.size();
    }

    vector<pair<vector<int16_t>,float>> toVector() {
        return vector<pair<vector<int16_t>,float>>(deq.begin(),deq.end());
    }

private:
    deque<pair<vector<int16_t>,float>> deq;
};

#endif // UTILS_H
