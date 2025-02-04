#ifndef RANDOM_H
#define RANDOM_H

#include <cstdint>
#include <cstdlib>

using namespace std;

static const double NORM_DOUBLE = 1.0 / (1ULL << 53);
static const float NORM_FLOAT = 1.0f / (1ULL << 24);

class Random {
public:
    Random() {
        seed = 685404112622437557ULL * (uint64_t)this + 54321ULL;
        seed ^= 9876542111ULL * rand() + 9632587411000005ULL;
        if (seed == 0ULL) {
            seed = 3210123505555ULL;
        }
    }

    Random(uint64_t seed) {
        if (seed == 0ULL) {
            seed = 3210123505555ULL;
        }
        this->seed = seed;
    }

    uint64_t nextLong() {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1DULL;
    }

    uint32_t nextInt(int n) {
        uint32_t x = nextLong() >> 32;
        uint64_t m = uint64_t(x) * uint64_t(n);
        return m >> 32;
    }

    double nextDouble() {
        return (nextLong() >> 11) * NORM_DOUBLE;
    }

    double nextDouble(double mi, double ma) {
        return mi + nextDouble() * (ma - mi);
    }

    float nextFloat() {
        return (float)((nextLong() >> 40) * NORM_FLOAT);
    }

    float nextFloat(float mi, float ma) {
        return mi + nextFloat() * (ma - mi);
    }

private:
    uint64_t seed;
};

#endif // RANDOM_H
