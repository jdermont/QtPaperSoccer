#ifndef NETWORK_H
#define NETWORK_H

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include "random.h"
// x86 with GCC/Clang: AVX2 code paths are compiled with target("avx2") whatever the
// build flags, and picked at runtime (see NetworkDeep2Cached2Int16)
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#define PS_AVX2_DISPATCH 1
#define PS_TARGET_AVX2 __attribute__((target("avx2")))
#include <immintrin.h>
#endif

using namespace std;

#define TYPE_0 0

class Network {
public:
    const int inputs,hidden;
    int type = 0;
    vector<float> hiddenWeights;
    vector<float> outputWeights;

    vector<float> cacheScores;

    explicit Network(int inputs, int hidden) : inputs(inputs), hidden(hidden) {
        hiddenWeights.resize(inputs * hidden);
        outputWeights.resize(hidden);

        cacheScores.resize(64 * hidden);

        Random random;
        for (auto & h : hiddenWeights) {
            h = random.nextFloat(-0.05f,0.05f);
        }
        for (auto & o : outputWeights) {
            o = random.nextFloat(-0.05f,0.05f);
        }
    }

    virtual void setWeights(Network *other) {
        this->hiddenWeights = other->hiddenWeights;
        this->outputWeights = other->outputWeights;
    }

    virtual void load(string name) {
        string name1 = std::to_string(hidden)+"_"+name;
        cout << "loadNetwork " << name1 << endl;
        ifstream plik; plik.open(name1);
        {
            char result[sizeof(float) * hiddenWeights.size()];
            plik.read(result, sizeof(float) * hiddenWeights.size());
            memcpy(&hiddenWeights[0], result, sizeof(float) * hiddenWeights.size());
        }
        char result[sizeof(float) * outputWeights.size()];
        plik.read(result, sizeof(float) * outputWeights.size());
        memcpy(&outputWeights[0], result, sizeof(float) * outputWeights.size());
        plik.close();
    }

    virtual void load2(string name1) {
        cout << "load " << name1 << endl;
        ifstream plik; plik.open(name1,std::ios::binary);
        {
            char result[sizeof(float) * hiddenWeights.size()];
            plik.read(result, sizeof(float) * hiddenWeights.size());
            memcpy(&hiddenWeights[0], result, sizeof(float) * hiddenWeights.size());
        }
        char result[sizeof(float) * outputWeights.size()];
        plik.read(result, sizeof(float) * outputWeights.size());
        memcpy(&outputWeights[0], result, sizeof(float) * outputWeights.size());
        plik.close();
    }

    virtual void save(string name) {
        string name1 = std::to_string(hidden)+"_"+name;
        cout << "save " << name1 << endl;
        ofstream plik; plik.open(name1, ios::out | ios::binary);
        {
            char result[sizeof(float) * hiddenWeights.size()];
            memcpy(result, &hiddenWeights[0], sizeof(float) * hiddenWeights.size());
            plik.write(result, sizeof(float) * hiddenWeights.size());
        }
        char result[sizeof(float) * outputWeights.size()];
        memcpy(result, &outputWeights[0], sizeof(float) * outputWeights.size());
        plik.write(result, sizeof(float) * outputWeights.size());
        plik.close();
    }

    virtual void shrinkWeights(float gamma) {
        if (gamma < -1) return;
        for (auto & h : hiddenWeights) {
            if (h < -5.0f) h = -5.0f;
            if (h > 5.0f) h = 5.0f;
        }
        for (auto & h : outputWeights) {
            if (h < -5.0f) h = -5.0f;
            if (h > 5.0f) h = 5.0f;
        }
    }

    virtual void cacheScore(const vector<int> & indexes, int id) {
        for (int i=0; i < hidden; i++) {
            cacheScores[id * hidden + i] = 0;
        }
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                cacheScores[id * hidden + i] += hiddenWeights[index*hidden+i];
            }
        }
    }

    virtual float getScoreDiff(const vector<int> & indexesBase, const vector<int> & indexes, int id) {
        vector<float> scores(hidden);
        for (auto & index : indexesBase) {
            for (int i=0; i < hidden; i++) {
                scores[i] -= hiddenWeights[index*hidden+i];
            }
        }
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu(cacheScores[id * hidden + i] + scores[i]);
        }

        float output = 0.0f;
        for (int i=0; i < hidden; i++) {
            output += outputWeights[i] * scores[i];
        }

        return fast_tanh(output);
    }

    virtual float getScore(const vector<int> & indexes) {
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu(scores[i]);
        }

        float output = 0.0f;
        for (int i=0; i < hidden; i++) {
            output += outputWeights[i] * scores[i];
        }

        return fast_tanh(output);
    }

    float relu(float x) {
        if (x < 0) return 0.01f * x;
        return x;
    }

    float fast_tanh(float x) {
        if (x > 4.95f) return 1;
        if (x < -4.95f) return -1;
        float x2 = x * x;
        float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }
};

class NetworkDeep2 : public Network {
public:
    const int hidden2;
    vector<float> hiddenWeights2;

    explicit NetworkDeep2(int inputs, int hidden, int hidden2) : Network(inputs,hidden), hidden2(hidden2) {
        hiddenWeights.resize(inputs * hidden);
        hiddenWeights2.resize(hidden * hidden2);
        outputWeights.resize(hidden2);

        Random random;
        for (auto & h : hiddenWeights) {
            h = random.nextFloat(-0.05f,0.05f);
        }
        for (auto & h : hiddenWeights2) {
            h = random.nextFloat(-0.05f,0.05f);
        }
        for (auto & o : outputWeights) {
            o = random.nextFloat(-0.05f,0.05f);
        }
    }

    virtual void setWeights(Network *other) {
        this->hiddenWeights = other->hiddenWeights;
        this->hiddenWeights2 = ((NetworkDeep2*)other)->hiddenWeights2;
        this->outputWeights = other->outputWeights;
    }

    virtual void shrinkWeights(float gamma) {
        if (gamma < -1) return;
        for (auto & h : hiddenWeights) {
            if (h < -5) h = -5;
            if (h > 5) h = 5;
        }
        for (auto & h : hiddenWeights2) {
            if (h < -5) h = -5;
            if (h > 5) h = 5;
        }
        for (auto & h : outputWeights) {
            if (h < -5) h = -5;
            if (h > 5) h = 5;
        }
    }

    virtual void load(string name) {
        string name1 = name;//std::to_string(hidden)+"_"+name;
        cout << "loadNetworkDeep2 " << name1 << endl;
        ifstream plik; plik.open(name1,ios::binary);
        {
            vector<char> result(sizeof(float)*hiddenWeights.size());
            plik.read(&result[0], sizeof(float) * hiddenWeights.size());
            memcpy(&hiddenWeights[0], &result[0], sizeof(float) * hiddenWeights.size());
        }
        {
            vector<char> result(sizeof(float)*hiddenWeights2.size());
            plik.read(&result[0], sizeof(float) * hiddenWeights2.size());
            memcpy(&hiddenWeights2[0], &result[0], sizeof(float) * hiddenWeights2.size());
        }
        {
            vector<char> result(sizeof(float)*outputWeights.size());
            plik.read(&result[0], sizeof(float) * outputWeights.size());
            memcpy(&outputWeights[0], &result[0], sizeof(float) * outputWeights.size());
        }
        plik.close();
    }

    virtual void save(string name) {
        string name1 = std::to_string(hidden)+"_"+name;
        cout << "save " << name1 << endl;
        ofstream plik; plik.open(name1, ios::out | ios::binary);
        {
            vector<char> result(sizeof(float)*hiddenWeights.size());
            memcpy(&result[0], &hiddenWeights[0], sizeof(float) * hiddenWeights.size());
            plik.write(&result[0], sizeof(float) * hiddenWeights.size());
        }
        {
            char result[sizeof(float) * hiddenWeights2.size()];
            memcpy(result, &hiddenWeights2[0], sizeof(float) * hiddenWeights2.size());
            plik.write(result, sizeof(float) * hiddenWeights2.size());
        }
        char result[sizeof(float) * outputWeights.size()];
        memcpy(result, &outputWeights[0], sizeof(float) * outputWeights.size());
        plik.write(result, sizeof(float) * outputWeights.size());
        plik.close();
    }

    virtual float getScoreDiff(const vector<int> & indexesBase, const vector<int> & indexes, int id) {
        vector<float> scores(hidden);
        for (auto & index : indexesBase) {
            for (int i=0; i < hidden; i++) {
                scores[i] -= hiddenWeights[index*hidden+i];
            }
        }
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu2(cacheScores[id * hidden + i] + scores[i]);
        }

        float output = 0;
        for (int i=0; i < hidden2; i++) {
            float score = 0.0;
            for (int j=0; j < hidden; j++) {
                score += scores[j] * hiddenWeights2[i*hidden+j];
            }
            score = relu(score);
            output += outputWeights[i] * score;
        }

        return fast_tanh(output);
    }

    virtual float getScore(const vector<int> & indexes) {
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu2(scores[i]);
        }

        float output = 0;
        for (int i=0; i < hidden2; i++) {
            float score = 0.0;
            for (int j=0; j < hidden; j++) {
                score += scores[j] * hiddenWeights2[i*hidden+j];
            }
            score = relu(score);
            output += outputWeights[i] * score;
        }

        return fast_tanh(output);
    }

    float relu2(float x) {
        if (x < 0) return 0.01f * x;
        return x*x;
    }
};

class NetworkDeep2Cached2 : public NetworkDeep2 {
public:
    NetworkDeep2Cached2(int inputs, int hidden, int hidden2) : NetworkDeep2(inputs,hidden,hidden2) {
        cout << "NetworkDeep2Cached2" << endl;
        cache.resize(52);
        for (int i=0; i < 52; i++) {
            cache[i].resize(57*57*hidden);
        }
    }

    void generateCache() {
        cout << "generateCache" << endl;
        for (int s1=0; s1 < 57; s1++) {
            for (int s2=0; s2 < 57; s2++) {
                int k = s1 * 57 + s2;
                for (int i=0; i < hidden; i++) {
                    for (int j=0; j < 52; j++) {
                        cache[j][k*hidden+i] = hiddenWeights[(373+s1+57*(j*2))*hidden+i] + hiddenWeights[(373+s2+57*(j*2+1))*hidden+i];
                    }
                }
            }
        }
    }

    virtual float getScore(const vector<int> & indexes) {
        int c = 0;
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            if (index >= 373) {
                break;
            }
            c++;
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }

        for (int j=0; j < 52; j++) {
            int k = 57 * (indexes[c+2*j]-373-57*(j*2)) + (indexes[c+2*j+1]-373-57*(2*j+1));
            for (int i=0; i < hidden; i++)  {
                scores[i] += cache[j][k*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu2(scores[i]);
        }

        float output = 0;
        for (int i=0; i < hidden2; i++) {
            float score = 0.0;
            for (int j=0; j < hidden; j++) {
                score += scores[j] * hiddenWeights2[i*hidden+j];
            }
            score = relu(score);
            output += outputWeights[i] * score;
        }

        return fast_tanh(output);
    }

    virtual float getScoreDiff(const vector<int> & indexesBase, const vector<int> & indexes, int id) {
        static thread_local vector<float> scores(hidden);
        memset(&scores[0],0,sizeof(float)*hidden);
        for (auto & index : indexesBase) {
            for (int i=0; i < hidden; i++) {
                scores[i] -= hiddenWeights[index*hidden+i];
            }
        }
        int c = 0;
        for (auto & index : indexes) {
            if (index >= 373) {
                break;
            }
            c++;
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int j=0; j < 52; j++) {
            int k = 57 * (indexes[c+2*j]-373-57*(j*2)) + (indexes[c+2*j+1]-373-57*(2*j+1));
            for (int i=0; i < hidden; i++)  {
                scores[i] += cache[j][k*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu2(cacheScores[id * hidden + i] + scores[i]);
        }

        float output = 0;
        for (int i=0; i < hidden2; i++) {
            float score = 0.0;
            for (int j=0; j < hidden; j++) {
                score += scores[j] * hiddenWeights2[i*hidden+j];
            }
            score = relu(score);
            output += outputWeights[i] * score;
        }

        return fast_tanh(output);
    }

private:
    vector<vector<float>> cache;
};


class NetworkDeep2Cached2Int16 : public NetworkDeep2 {
public:
    float scale = 0;
    bool avx2 = false;

    NetworkDeep2Cached2Int16(int inputs, int hidden, int hidden2) : NetworkDeep2(inputs,hidden,hidden2) {
#ifdef PS_AVX2_DISPATCH
        __builtin_cpu_init();
        avx2 = __builtin_cpu_supports("avx2");
#endif
        cout << "NetworkDeep2Cached2Int16 " << (avx2 ? "avx2" : "plain") << endl;
    }

    void generateCache() {
        cout << "generateCache" << endl;
        scales.assign(hidden, scale);
        if (scale <= 0) {
            for (int i=0; i < hidden; i++) {
                float b = 0;
                for (int j=0; j < 52; j++) {
                    float hi = 0, lo = 0;
                    for (int k=0; k < 2; k++) {
                        float mx = -1e30f, mn = 1e30f;
                        for (int t=0; t < 57; t++) {
                            float w = hiddenWeights[(373+57*(2*j+k)+t)*hidden+i];
                            mx = max(mx, w); mn = min(mn, w);
                        }
                        hi += mx; lo += mn;
                    }
                    b = max(b, max(fabs(hi), fabs(lo)));
                }
                scales[i] = 32767 / max(b, 1e-6f);
            }
        }
        invScales.resize(hidden);
        for (int i=0; i < hidden; i++) invScales[i] = 1.0f / scales[i];
        cache.assign((size_t)52*57*57*hidden, 0);
        for (int j=0; j < 52; j++) {
            const float *w0 = &hiddenWeights[(373+57*(2*j))*hidden];
            const float *w1 = &hiddenWeights[(373+57*(2*j+1))*hidden];
            for (int s1=0; s1 < 57; s1++) {
                for (int s2=0; s2 < 57; s2++) {
                    int16_t *out = &cache[((size_t)j*57*57 + s1*57 + s2) * hidden];
                    for (int i=0; i < hidden; i++) {
                        float q = roundf((w0[s1*hidden+i] + w1[s2*hidden+i]) * scales[i]);
                        out[i] = (int16_t)max(-32767.0f, min(32767.0f, q));
                    }
                }
            }
        }
    }

    virtual float getScore(const vector<int> & indexes) override {
        return evaluate({}, indexes, nullptr);
    }

    virtual float getScoreDiff(const vector<int> & indexesBase, const vector<int> & indexes, int id) override {
        return evaluate(indexesBase, indexes, &cacheScores[id * hidden]);
    }

private:
    vector<float> scales, invScales;
    vector<int16_t> cache;

    float evaluate(const vector<int> & indexesBase, const vector<int> & indexes, const float *base) {
        const int h = hidden;
        static thread_local vector<float> scoresBuffer;
        scoresBuffer.resize(h);
        float *scores = scoresBuffer.data();

        if (base) memcpy(scores, base, h * sizeof(float));
        else memset(scores, 0, h * sizeof(float));
        for (int idx : indexesBase) addRowAny(scores, &hiddenWeights[idx*h], h, -1.0f);
        int c = 0;
        for (int idx : indexes) {
            if (idx >= 373) break;
            addRowAny(scores, &hiddenWeights[idx*h], h, 1.0f);
            c++;
        }
        const int16_t *rows[52];
        for (int j=0; j < 52; j++) {
            int k = 57 * (indexes[c+2*j]-373-57*(j*2)) + (indexes[c+2*j+1]-373-57*(2*j+1));
            rows[j] = &cache[((size_t)j*57*57 + k) * h];
        }
#ifdef PS_AVX2_DISPATCH
        if (avx2) return finishAvx2(rows, scores);
#endif
        return finishPlain(rows, scores);
    }

    float finishPlain(const int16_t *const *rows, float *scores) {
        const int h = hidden;
        const float *inv = invScales.data();
        static thread_local vector<int32_t> accBuffer;
        accBuffer.assign(h, 0);
        int32_t *acc = accBuffer.data();
        for (int j=0; j < 52; j++) {
            const int16_t *w = rows[j];
            for (int i=0; i < h; i++) acc[i] += w[i];
        }
        for (int i=0; i < h; i++) scores[i] = relu2(scores[i] + acc[i] * inv[i]);

        float output = 0;
        for (int o=0; o < hidden2; o++) {
            const float *w = &hiddenWeights2[o*h];
            float d = 0;
            for (int i=0; i < h; i++) d += scores[i] * w[i];
            output += outputWeights[o] * relu(d);
        }
        return fast_tanh(output);
    }

    static void addRow(float *scores, const float *w, int h, float sign) {
        for (int i=0; i < h; i++) scores[i] += sign * w[i];
    }

    void addRowAny(float *scores, const float *w, int h, float sign) {
#ifdef PS_AVX2_DISPATCH
        if (avx2) { addRowAvx2(scores, w, h, sign); return; }
#endif
        addRow(scores, w, h, sign);
    }

#ifdef PS_AVX2_DISPATCH

    PS_TARGET_AVX2 float finishAvx2(const int16_t *const *rows, float *scores) {
        const int h = hidden;
        const float *inv = invScales.data();
        int i = 0;
        for (; i + 32 <= h; i += 32) accumulate<4>(rows, i, inv, scores);
        for (; i + 8 <= h; i += 8) accumulate<1>(rows, i, inv, scores);
        for (; i < h; i++) {
            int32_t s = 0;
            for (int j=0; j < 52; j++) s += rows[j][i];
            scores[i] = relu2(scores[i] + s * inv[i]);
        }

        float output = 0;
        for (int o=0; o < hidden2; o++) {
            const float *w = &hiddenWeights2[o*h];
            __m256 sum = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= h; j += 8) sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(scores + j), _mm256_loadu_ps(w + j)));
            float d = hsum256(sum);
            for (; j < h; j++) d += scores[j] * w[j];
            output += outputWeights[o] * relu(d);
        }
        return fast_tanh(output);
    }

    PS_TARGET_AVX2 static void addRowAvx2(float *scores, const float *w, int h, float sign) {
        __m256 s = _mm256_set1_ps(sign);
        int i = 0;
        for (; i + 8 <= h; i += 8) {
            _mm256_storeu_ps(scores + i, _mm256_add_ps(_mm256_loadu_ps(scores + i), _mm256_mul_ps(s, _mm256_loadu_ps(w + i))));
        }
        for (; i < h; i++) scores[i] += sign * w[i];
    }

    PS_TARGET_AVX2 static inline float hsum256(__m256 v) {
        __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        x = _mm_add_ps(x, _mm_movehl_ps(x, x));
        x = _mm_add_ss(x, _mm_movehdup_ps(x));
        return _mm_cvtss_f32(x);
    }

    PS_TARGET_AVX2 static inline __m256 relu2_avx(__m256 x) {
        __m256 neg = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
        return _mm256_blendv_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(x, _mm256_set1_ps(0.01f)), neg);
    }

    template <int R>
    PS_TARGET_AVX2 static void accumulate(const int16_t *const *rows, int offset, const float *inv, float *scores) {
        __m256i acc[R];
        for (int r = 0; r < R; r++) acc[r] = _mm256_setzero_si256();
        for (int k = 0; k < 52; k++) {
            const int16_t *w = rows[k] + offset;
            for (int r = 0; r < R; r++) {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(w + 8*r))));
            }
        }
        for (int r = 0; r < R; r++) {
            int o = offset + 8*r;
            __m256 x = _mm256_add_ps(_mm256_loadu_ps(scores + o), _mm256_mul_ps(_mm256_cvtepi32_ps(acc[r]), _mm256_loadu_ps(inv + o)));
            _mm256_storeu_ps(scores + o, relu2_avx(x));
        }
    }
#endif
};


#endif // NETWORK_H
