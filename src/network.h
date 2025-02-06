#ifndef NETWORK_H
#define NETWORK_H

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include "random.h"

using namespace std;

#define TYPE_0 0

struct QNetwork {
    vector<float> hiddenWeights;
    vector<float> outputWeights;

    vector<float> hiddenMomentum;
    vector<float> outputMomentum;

    const int inputs;
    const int hidden;
    const int outputs;

    float alpha = 0.001f;
    float mom = 0.8f;

    explicit QNetwork(int inputs, int hidden, int outputs, bool initWeights = false) : inputs(inputs), hidden(hidden), outputs(outputs) {
        hiddenWeights.resize(hidden*inputs);
        outputWeights.resize(hidden*outputs);
        hiddenMomentum.resize(hidden*inputs);
        outputMomentum.resize(hidden*outputs);

        if (initWeights) {
            Random random;
            for (auto & h : hiddenWeights) {
                h = random.nextFloat(-0.03f,0.03f);
            }
            for (auto & o : outputWeights) {
                o = random.nextFloat(-0.001f,0.001f);
            }
        }
    }

    void multiply(float m) {
        for (auto & h : hiddenWeights) {
            h *= m;
        }
        for (auto & h : outputWeights) {
            h *= m;
        }
    }

    void clip(float ma) {
        float c = 0;
        for (auto & h : hiddenWeights) {
            if (abs(h) > ma) {
                if (abs(h) > c) {
                    c = abs(h);
                }
            }
        }
        for (auto & h : outputWeights) {
            if (abs(h) > ma) {
                if (abs(h) > c) {
                    c = abs(h);
                }
            }
        }
        if (c > 0) {
            float ratio = ma / c;
            for (auto & h : hiddenWeights) {
                h *= ratio;
            }
            for (auto & h : outputWeights) {
                h *= ratio;
            }
        }
    }

    void load(string name) {
        ifstream plik;
        plik.open(name);
        const int size = inputs * hidden + hidden * outputs;
        char result[1<<20];
        plik.read(result, sizeof(float) * size);
        vector<float> solution(size);
        memcpy(&solution[0], result, sizeof(float) * size);
        fromVector(solution);
        plik.close();
    }

    void save(string name) {
        string name1 = std::to_string(hidden)+"_"+name;
        cout << "save " << name1 << endl;
        ofstream plik; plik.open(name1, ios::out | ios::binary);
        {
            char result[sizeof(float) * inputs * hidden];
            memcpy(result, &hiddenWeights[0], sizeof(float) * inputs * hidden);
            plik.write(result, sizeof(float) * inputs * hidden);
        }
        {
            char result[sizeof(float) * outputs * hidden];
            memcpy(result, &outputWeights[0], sizeof(float) * outputs * hidden);
            plik.write(result, sizeof(float) * outputs * hidden);
        }
        plik.close();
    }

    void setWeights(const QNetwork & other) {
        this->hiddenWeights = other.hiddenWeights;
        this->outputWeights = other.outputWeights;
    }

    virtual void fromVector(const vector<float> & v) {
        memcpy(&hiddenWeights[0],&v[0],sizeof(float) * hidden * inputs);
        memcpy(&outputWeights[0],&v[hidden * inputs],sizeof(float) * outputs * hidden);
    }


    float relu(float x) {
        if (x < 0) return 0.01f * x;
        return x;
    }

    float drelu(float x) {
        if (x < 0) return 0.01f;
        return 1;
    }

    vector<float> predict(const vector<int> & ts) {
        vector<float> scores(hidden);

        for (auto & index : ts) {
            for (int j=0; j < hidden; j++) {
                scores[j] += hiddenWeights[index*hidden+j];
            }
        }

        for (int i=0; i < hidden; i++) {
            scores[i] = relu(scores[i]);
        }

        vector<float> ops(outputs);
        for (int i=0; i < outputs; i++) {
            float op = 0;
            for (int j=0; j < hidden; j++) {
                op += outputWeights[i*hidden+j] * scores[j];
            }
            ops[i] = op;
        }

        return ops;
    }

    vector<float> predict(const vector<int> & ts, int t) {
        vector<float> scores(hidden);

        for (auto & index : ts) {
            for (int j=0; j < hidden; j++) {
                scores[j] += hiddenWeights[index*hidden+j];
            }
        }

        for (int i=0; i < hidden; i++) {
            scores[i] = relu(scores[i]);
        }

        vector<float> ops(outputs);
        for (int i=t; i < t+1; i++) {
            float op = 0;
            for (int j=0; j < hidden; j++) {
                op += outputWeights[i*hidden+j] * scores[j];
            }
            ops[i] = op;
        }

        return ops;
    }

    float learn(const vector<int> & ts, int action, float target) {
        vector<float> scores(hidden);
        for (auto & index : ts) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }

        for (int i=0; i < hidden; i++) {
            scores[i] = relu(scores[i]);
        }

        vector<float> ops(outputs);

        for (int i=action; i < action+1; i++) {
            float op = 0;
            for (int j=0; j < hidden; j++) {
                op += outputWeights[i*hidden+j] * scores[j];
            }
            ops[i] = op;
        }

        vector<float> errors(outputs);
        for (int i=action; i < action+1; i++) {
            errors[i] = (target - ops[i]) * 1;
            if (errors[i] > 1) errors[i] = 1;
            if (errors[i] < -1) errors[i] = -1;
        }


        vector<float> eh(hidden);
        for (int i=action; i < action+1; i++) {
            for (int j=0; j < hidden; j++) {
                eh[j] += errors[i] * outputWeights[i*hidden+j] * drelu(scores[j]);
            }
        }

        for (int i=action; i < action+1; i++) {
            for (int j=0; j < hidden; j++) {
                outputMomentum[i*hidden+j] = mom * outputMomentum[i*hidden+j] + alpha * scores[j] * errors[i];
                outputWeights[i*hidden+j] += outputMomentum[i*hidden+j];
            }
        }

        for (auto & index : ts)  {
            for (int i=0; i < hidden; i++) {
                hiddenMomentum[index*hidden+i] = mom * hiddenMomentum[index*hidden+i] + alpha * eh[i];
                hiddenWeights[index*hidden+i] += hiddenMomentum[index*hidden+i];
            }
        }

        return abs(errors[action]);
    }
};


class Network {
public:
    const int inputs,hidden;
    int type = 0;
    vector<float> hiddenWeights;
    vector<float> aHiddenWeights;
    vector<float> nHiddenWeights;
    vector<float> hiddenMomentum;
    vector<float> outputWeights;
    vector<float> aOutputWeights;
    vector<float> nOutputWeights;
    vector<float> outputMomentum;

    vector<float> cacheScores;

    float alpha = 0.001f;
    float mom = 0.8f;

    explicit Network(int inputs, int hidden) : inputs(inputs), hidden(hidden) {
        hiddenWeights.resize(inputs * hidden);
        aHiddenWeights.resize(inputs * hidden);
        nHiddenWeights.resize(inputs * hidden);
        hiddenMomentum.resize(inputs * hidden);
        outputWeights.resize(hidden);
        aOutputWeights.resize(hidden);
        nOutputWeights.resize(hidden);
        outputMomentum.resize(hidden);

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
        // name1 = "/home/jacek/1projekty/PaperSoccer/96_RL9";
        // cout << "load " << name1 << endl;
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

    virtual void loadCheckpoint(string name) {
        string name1 = "_"+std::to_string(hidden)+"_"+name;
        cout << "load checkpoint " << name1 << endl;
        ifstream plik; plik.open(name1);
        {
            char result[sizeof(float) * hiddenWeights.size()];
            plik.read(result, sizeof(float) * hiddenWeights.size());
            memcpy(&hiddenWeights[0], result, sizeof(float) * hiddenWeights.size());
        }
        {
            char result[sizeof(float) * hiddenMomentum.size()];
            plik.read(result, sizeof(float) * hiddenMomentum.size());
            memcpy(&hiddenMomentum[0], result, sizeof(float) * hiddenMomentum.size());
        }
        {
            char result[sizeof(float) * aHiddenWeights.size()];
            plik.read(result, sizeof(float) * aHiddenWeights.size());
            memcpy(&aHiddenWeights[0], result, sizeof(float) * aHiddenWeights.size());
        }
        {
            char result[sizeof(float) * nHiddenWeights.size()];
            plik.read(result, sizeof(float) * nHiddenWeights.size());
            memcpy(&nHiddenWeights[0], result, sizeof(float) * nHiddenWeights.size());
        }
        {
            char result[sizeof(float) * outputWeights.size()];
            plik.read(result, sizeof(float) * outputWeights.size());
            memcpy(&outputWeights[0], result, sizeof(float) * outputWeights.size());
        }
        {
            char result[sizeof(float) * outputMomentum.size()];
            plik.read(result, sizeof(float) * outputMomentum.size());
            memcpy(&outputMomentum[0], result, sizeof(float) * outputMomentum.size());
        }
        {
            char result[sizeof(float) * aOutputWeights.size()];
            plik.read(result, sizeof(float) * aOutputWeights.size());
            memcpy(&aOutputWeights[0], result, sizeof(float) * aOutputWeights.size());
        }
        {
            char result[sizeof(float) * nOutputWeights.size()];
            plik.read(result, sizeof(float) * nOutputWeights.size());
            memcpy(&nOutputWeights[0], result, sizeof(float) * nOutputWeights.size());
        }

        plik.close();
    }

    virtual void saveCheckpoint(string name) {
        string name1 = "_"+std::to_string(hidden)+"_"+name;
        cout << "save checkpoint " << name1 << endl;
        ofstream plik; plik.open(name1, ios::out | ios::binary);
        {
            char result[sizeof(float) * hiddenWeights.size()];
            memcpy(result, &hiddenWeights[0], sizeof(float) * hiddenWeights.size());
            plik.write(result, sizeof(float) * hiddenWeights.size());
        }
        {
            char result[sizeof(float) * hiddenMomentum.size()];
            memcpy(result, &hiddenMomentum[0], sizeof(float) * hiddenMomentum.size());
            plik.write(result, sizeof(float) * hiddenMomentum.size());
        }
        {
            char result[sizeof(float) * aHiddenWeights.size()];
            memcpy(result, &aHiddenWeights[0], sizeof(float) * aHiddenWeights.size());
            plik.write(result, sizeof(float) * aHiddenWeights.size());
        }
        {
            char result[sizeof(float) * nHiddenWeights.size()];
            memcpy(result, &nHiddenWeights[0], sizeof(float) * nHiddenWeights.size());
            plik.write(result, sizeof(float) * nHiddenWeights.size());
        }
        {
            char result[sizeof(float) * outputWeights.size()];
            memcpy(result, &outputWeights[0], sizeof(float) * outputWeights.size());
            plik.write(result, sizeof(float) * outputWeights.size());
        }
        {
            char result[sizeof(float) * outputMomentum.size()];
            memcpy(result, &outputMomentum[0], sizeof(float) * outputMomentum.size());
            plik.write(result, sizeof(float) * outputMomentum.size());
        }
        {
            char result[sizeof(float) * aOutputWeights.size()];
            memcpy(result, &aOutputWeights[0], sizeof(float) * aOutputWeights.size());
            plik.write(result, sizeof(float) * aOutputWeights.size());
        }
        {
            char result[sizeof(float) * nOutputWeights.size()];
            memcpy(result, &nOutputWeights[0], sizeof(float) * nOutputWeights.size());
            plik.write(result, sizeof(float) * nOutputWeights.size());
        }
        plik.close();
    }

    virtual void shrinkWeights(float gamma) {
        // float ma = 0;
        // for (auto & h : hiddenWeights) {
        //     h *= gamma;
        //     ma = max(ma,abs(h));
        // }
        // for (auto & o : outputWeights) {
        //     o *= gamma;
        //     ma = max(ma,abs(o));
        // }
        // if (ma > 5) {
        //     float ratio = 5 / ma;
        //     for (auto & h : hiddenWeights) {
        //         h *= ratio;
        //     }
        //     for (auto & o : outputWeights) {
        //         o *= ratio;
        //     }
        // }
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

    virtual void learn(const vector<int> & indexes, float target) {
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu(scores[i]);
        }

        float op = 0.0;
        for (int i=0; i < hidden; i++) {
            op += outputWeights[i] * scores[i];
        }
        op = fast_tanh(op);

        float e = target - op;
        float dop = e * dtanh(op);

        vector<float> eh(hidden);
        for (int i=0; i < hidden; i++) {
            eh[i] = dop * outputWeights[i];
        }

        vector<float> dh(hidden);
        for (int i=0; i < hidden; i++) {
            dh[i] = eh[i] * drelu(scores[i]);
        }

        for (int i=0; i < hidden; i++) {
            outputMomentum[i] = mom * outputMomentum[i] + alpha * scores[i] * dop;
            outputWeights[i] += outputMomentum[i];
        }

        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                hiddenMomentum[index*hidden+i] = mom * hiddenMomentum[index*hidden+i] + alpha * dh[i];
                hiddenWeights[index*hidden+i] += hiddenMomentum[index*hidden+i];
            }
        }
    }

    virtual void learnRL(const vector<int> & indexes, float target) {
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu(scores[i]);
        }

        float op = 0.0;
        for (int i=0; i < hidden; i++) {
            op += outputWeights[i] * scores[i];
        }
        op = fast_tanh(op);

        float e = target - op;
        float dop = e * dtanh(op);

        vector<float> eh(hidden);
        for (int i=0; i < hidden; i++) {
            eh[i] = dop * outputWeights[i];
        }

        vector<float> dh(hidden);
        for (int i=0; i < hidden; i++) {
            dh[i] = eh[i] * drelu(scores[i]);
        }

        for (int i=0; i < hidden; i++) {
            float a = aOutputWeights[i] == 0 ? 1 : (abs(nOutputWeights[i]) / aOutputWeights[i]);
            outputWeights[i] += 0.2 * a * scores[i] * dop;
            aOutputWeights[i] += abs(scores[i] * dop);
            nOutputWeights[i] += scores[i] * dop;
        }

        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                float a = aHiddenWeights[index*hidden+i] == 0 ? 1 : (abs(nHiddenWeights[index*hidden+i]) / aHiddenWeights[index*hidden+i]);
                hiddenWeights[index*hidden+i] += 0.2 * a * dh[i];
                aHiddenWeights[index*hidden+i] += abs(dh[i]);
                nHiddenWeights[index*hidden+i] += dh[i];
            }
        }
    }

    void print(float a, float d, string name) {
        float m = 0;
        float sum = 0;
        for (auto & i : hiddenWeights) {
            if (abs(i) > m) m = abs(i);
            int32_t s = (int32_t)round((d*(i+a)));
            float x = (s/d)-a;
            sum += x;
        }
        for (auto & i : outputWeights) {
            if (abs(i) > m) m = abs(i);
            int32_t s = (int32_t)round((d*(i+a)));
            float x = (s/d)-a;
            sum += x;
        }

        stringstream ss;
        for (auto & i : hiddenWeights) {
            int32_t s = (int32_t)round(d*(i+a));
            char a = (char)(s&255);
            char b = (char)((s>>8)&255);
            ss << b << a;
        }
        for (auto & i : outputWeights) {
            int32_t s = (int32_t)round(d*(i+a));
            char a = (char)(s&255);
            char b = (char)((s>>8)&255);
            ss << b << a;
        }

        cout << "max " << m << endl;
        cout << "sum " << sum << endl;
        ofstream plik; plik.open(name);
        plik << ss.str();
        plik.close();
    }

    void print2(float a, float d, string name) {
        float m = 0;
        float sum = 0;
        uint64_t suma = 0;
        int sMin = 100000;
        int sMax = 0;
        for (size_t i=0; i < hiddenWeights.size(); i++) {
            float h = hiddenWeights[i];
            if (h > a) h = a;
            if (h < -a) h = -a;
            int32_t s = (int32_t)round((d*(h+a)));
            float x = (s/d)-a;
            sum += x;
            if (abs(x) > m) m = abs(x);
        }
        for (size_t i=0; i < outputWeights.size(); i++) {
            float h = outputWeights[i];
            if (h > a) h = a;
            if (h < -a) h = -a;
            int32_t s = (int32_t)round((d*(h+a)));
            float x = (s/d)-a;
            sum += x;
            if (abs(x) > m) m = abs(x);
        }

        stringstream ss;
        for (size_t i=0; i < hiddenWeights.size(); i+=2) {
            float h = hiddenWeights[i];
            if (h > a) h = a;
            if (h < -a) h = -a;
            int32_t s = (int32_t)round((d*(h+a)));
            float h2 = hiddenWeights[i+1];
            if (h2 > a) h2 = a;
            if (h2 < -a) h2 = -a;
            int32_t s2 = (int32_t)round((d*(h2+a)));
            int32_t s3 = s * 234 + s2;
            sMin = min(s3,sMin);
            sMax = max(s3,sMax);
            suma += s3;
            char a = (char)(s3&255);
            char b = (char)((s3>>8)&255);
            ss << b << a;
        }
        for (size_t i=0; i < outputWeights.size(); i+=2) {
            float h = outputWeights[i];
            if (h > a) h = a;
            if (h < -a) h = -a;
            int32_t s = (int32_t)round((d*(h+a)));
            float h2 = outputWeights[i+1];
            if (h2 > a) h2 = a;
            if (h2 < -a) h2 = -a;
            int32_t s2 = (int32_t)round((d*(h2+a)));
            int32_t s3 = s * 234 + s2;
            sMin = min(s3,sMin);
            sMax = max(s3,sMax);
            suma += s3;
            char a = (char)(s3&255);
            char b = (char)((s3>>8)&255);
            ss << b << a;
        }

        cout << "max " << m << endl;
        cout << "sum " << sum << endl;
        cout << "suma " << suma << endl;
        cout << "sMin " << sMin << endl;
        cout << "sMax " << sMax << endl;
        ofstream plik; plik.open(name);
        plik << ss.str();
        plik.close();
    }

    float relu(float x) {
        if (x < 0) return 0.01f * x;
        return x;
    }

    float drelu(float x) {
        if (x < 0) return 0.01f;
        return 1;
    }

    float dtanh(float x) {
        float t = 1 - x*x;
        if (t < 0.03f) t = 0.03f;
        return t;
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

class NetworkScrelu : public Network {
public:
    explicit NetworkScrelu(int inputs, int hidden) : Network(inputs,hidden) {

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
            scores[i] = relu2(scores[i]);
        }

        float output = 0.0f;
        for (int i=0; i < hidden; i++) {
            output += outputWeights[i] * scores[i];
        }

        return fast_tanh(output);
    }

    virtual void learn(const vector<int> & indexes, float target) {
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu2(scores[i]);
        }

        float op = 0.0;
        for (int i=0; i < hidden; i++) {
            op += outputWeights[i] * scores[i];
        }
        op = fast_tanh(op);

        float e = target - op;
        float dop = e * dtanh(op);

        vector<float> eh(hidden);
        for (int i=0; i < hidden; i++) {
            eh[i] = dop * outputWeights[i];
        }

        vector<float> dh(hidden);
        for (int i=0; i < hidden; i++) {
            dh[i] = eh[i] * drelu2(scores[i]);
        }

        for (int i=0; i < hidden; i++) {
            outputMomentum[i] = mom * outputMomentum[i] + alpha * scores[i] * dop;
            outputWeights[i] += outputMomentum[i];
        }

        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                hiddenMomentum[index*hidden+i] = mom * hiddenMomentum[index*hidden+i] + alpha * dh[i];
                hiddenWeights[index*hidden+i] += hiddenMomentum[index*hidden+i];
            }
        }
    }

    virtual void learnRL(const vector<int> & indexes, float target) {
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu2(scores[i]);
        }

        float op = 0.0;
        for (int i=0; i < hidden; i++) {
            op += outputWeights[i] * scores[i];
        }
        op = fast_tanh(op);

        float e = target - op;
        float dop = e * dtanh(op);

        vector<float> eh(hidden);
        for (int i=0; i < hidden; i++) {
            eh[i] = dop * outputWeights[i];
        }

        vector<float> dh(hidden);
        for (int i=0; i < hidden; i++) {
            dh[i] = eh[i] * drelu2(scores[i]);
        }

        for (int i=0; i < hidden; i++) {
            float a = aOutputWeights[i] == 0 ? 1 : (abs(nOutputWeights[i]) / aOutputWeights[i]);
            outputWeights[i] += 0.2 * a * scores[i] * dop;
            aOutputWeights[i] += abs(scores[i] * dop);
            nOutputWeights[i] += scores[i] * dop;
        }

        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                float a = aHiddenWeights[index*hidden+i] == 0 ? 1 : (abs(nHiddenWeights[index*hidden+i]) / aHiddenWeights[index*hidden+i]);
                hiddenWeights[index*hidden+i] += 0.2 * a * dh[i];
                aHiddenWeights[index*hidden+i] += abs(dh[i]);
                nHiddenWeights[index*hidden+i] += dh[i];
            }
        }
    }

    float relu2(float x) {
        if (x < 0) return 0.01f * x;
        if (x > 1) return 0.99f + 0.01f * x;
        return x*x;
    }

    float drelu2(float x) {
        if (x < 0 || x > 1) return 0.01f;
        return 2 * sqrtf(x);
    }
};

class NetworkDeep : public Network {
public:
    const int hidden2;
    vector<float> hiddenWeights2;
    vector<float> aHiddenWeights2;
    vector<float> nHiddenWeights2;
    vector<float> hiddenMomentum2;

    explicit NetworkDeep(int inputs, int hidden, int hidden2) : Network(inputs,hidden), hidden2(hidden2) {
        // hiddenWeights.resize(inputs * hidden);
        // aHiddenWeights.resize(inputs * hidden);
        // nHiddenWeights.resize(inputs * hidden);
        // hiddenMomentum.resize(inputs * hidden);
        hiddenWeights2.resize(hidden * hidden2);
        aHiddenWeights2.resize(hidden * hidden2);
        nHiddenWeights2.resize(hidden * hidden2);
        hiddenMomentum2.resize(hidden * hidden2);
        outputWeights.resize(hidden2);
        aOutputWeights.resize(hidden2);
        nOutputWeights.resize(hidden2);
        outputMomentum.resize(hidden2);

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
        this->hiddenWeights2 = ((NetworkDeep*)other)->hiddenWeights2;
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

    virtual void load(string name1) {
        cout << "load " << name1 << endl;
        ifstream plik; plik.open(name1,ios::binary);
        {
            char result[sizeof(float) * hiddenWeights.size()];
            plik.read(result, sizeof(float) * hiddenWeights.size());
            memcpy(&hiddenWeights[0], result, sizeof(float) * hiddenWeights.size());
        }
        {
            char result[sizeof(float) * hiddenWeights2.size()];
            plik.read(result, sizeof(float) * hiddenWeights2.size());
            memcpy(&hiddenWeights2[0], result, sizeof(float) * hiddenWeights2.size());
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

    virtual void loadCheckpoint(string name) {
        string name1 = "_"+std::to_string(hidden)+"_"+name;
        cout << "load checkpoint " << name1 << endl;
        ifstream plik; plik.open(name1);
        {
            char result[sizeof(float) * hiddenWeights.size()];
            plik.read(result, sizeof(float) * hiddenWeights.size());
            memcpy(&hiddenWeights[0], result, sizeof(float) * hiddenWeights.size());
        }
        {
            char result[sizeof(float) * hiddenMomentum.size()];
            plik.read(result, sizeof(float) * hiddenMomentum.size());
            memcpy(&hiddenMomentum[0], result, sizeof(float) * hiddenMomentum.size());
        }
        {
            char result[sizeof(float) * aHiddenWeights.size()];
            plik.read(result, sizeof(float) * aHiddenWeights.size());
            memcpy(&aHiddenWeights[0], result, sizeof(float) * aHiddenWeights.size());
        }
        {
            char result[sizeof(float) * nHiddenWeights.size()];
            plik.read(result, sizeof(float) * nHiddenWeights.size());
            memcpy(&nHiddenWeights[0], result, sizeof(float) * nHiddenWeights.size());
        }
        {
            char result[sizeof(float) * hiddenWeights2.size()];
            plik.read(result, sizeof(float) * hiddenWeights2.size());
            memcpy(&hiddenWeights2[0], result, sizeof(float) * hiddenWeights2.size());
        }
        {
            char result[sizeof(float) * hiddenMomentum2.size()];
            plik.read(result, sizeof(float) * hiddenMomentum2.size());
            memcpy(&hiddenMomentum2[0], result, sizeof(float) * hiddenMomentum2.size());
        }
        {
            char result[sizeof(float) * aHiddenWeights2.size()];
            plik.read(result, sizeof(float) * aHiddenWeights2.size());
            memcpy(&aHiddenWeights2[0], result, sizeof(float) * aHiddenWeights2.size());
        }
        {
            char result[sizeof(float) * nHiddenWeights2.size()];
            plik.read(result, sizeof(float) * nHiddenWeights2.size());
            memcpy(&nHiddenWeights2[0], result, sizeof(float) * nHiddenWeights2.size());
        }
        {
            char result[sizeof(float) * outputWeights.size()];
            plik.read(result, sizeof(float) * outputWeights.size());
            memcpy(&outputWeights[0], result, sizeof(float) * outputWeights.size());
        }
        {
            char result[sizeof(float) * outputMomentum.size()];
            plik.read(result, sizeof(float) * outputMomentum.size());
            memcpy(&outputMomentum[0], result, sizeof(float) * outputMomentum.size());
        }
        {
            char result[sizeof(float) * aOutputWeights.size()];
            plik.read(result, sizeof(float) * aOutputWeights.size());
            memcpy(&aOutputWeights[0], result, sizeof(float) * aOutputWeights.size());
        }
        {
            char result[sizeof(float) * nOutputWeights.size()];
            plik.read(result, sizeof(float) * nOutputWeights.size());
            memcpy(&nOutputWeights[0], result, sizeof(float) * nOutputWeights.size());
        }

        plik.close();
    }

    virtual void saveCheckpoint(string name) {
        string name1 = "_"+std::to_string(hidden)+"_"+name;
        cout << "save checkpoint " << name1 << endl;
        ofstream plik; plik.open(name1, ios::out | ios::binary);
        {
            char result[sizeof(float) * hiddenWeights.size()];
            memcpy(result, &hiddenWeights[0], sizeof(float) * hiddenWeights.size());
            plik.write(result, sizeof(float) * hiddenWeights.size());
        }
        {
            char result[sizeof(float) * hiddenMomentum.size()];
            memcpy(result, &hiddenMomentum[0], sizeof(float) * hiddenMomentum.size());
            plik.write(result, sizeof(float) * hiddenMomentum.size());
        }
        {
            char result[sizeof(float) * aHiddenWeights.size()];
            memcpy(result, &aHiddenWeights[0], sizeof(float) * aHiddenWeights.size());
            plik.write(result, sizeof(float) * aHiddenWeights.size());
        }
        {
            char result[sizeof(float) * nHiddenWeights.size()];
            memcpy(result, &nHiddenWeights[0], sizeof(float) * nHiddenWeights.size());
            plik.write(result, sizeof(float) * nHiddenWeights.size());
        }
        {
            char result[sizeof(float) * hiddenWeights2.size()];
            memcpy(result, &hiddenWeights2[0], sizeof(float) * hiddenWeights2.size());
            plik.write(result, sizeof(float) * hiddenWeights2.size());
        }
        {
            char result[sizeof(float) * hiddenMomentum2.size()];
            memcpy(result, &hiddenMomentum2[0], sizeof(float) * hiddenMomentum2.size());
            plik.write(result, sizeof(float) * hiddenMomentum2.size());
        }
        {
            char result[sizeof(float) * aHiddenWeights2.size()];
            memcpy(result, &aHiddenWeights2[0], sizeof(float) * aHiddenWeights2.size());
            plik.write(result, sizeof(float) * aHiddenWeights2.size());
        }
        {
            char result[sizeof(float) * nHiddenWeights2.size()];
            memcpy(result, &nHiddenWeights2[0], sizeof(float) * nHiddenWeights2.size());
            plik.write(result, sizeof(float) * nHiddenWeights2.size());
        }
        {
            char result[sizeof(float) * outputWeights.size()];
            memcpy(result, &outputWeights[0], sizeof(float) * outputWeights.size());
            plik.write(result, sizeof(float) * outputWeights.size());
        }
        {
            char result[sizeof(float) * outputMomentum.size()];
            memcpy(result, &outputMomentum[0], sizeof(float) * outputMomentum.size());
            plik.write(result, sizeof(float) * outputMomentum.size());
        }
        {
            char result[sizeof(float) * aOutputWeights.size()];
            memcpy(result, &aOutputWeights[0], sizeof(float) * aOutputWeights.size());
            plik.write(result, sizeof(float) * aOutputWeights.size());
        }
        {
            char result[sizeof(float) * nOutputWeights.size()];
            memcpy(result, &nOutputWeights[0], sizeof(float) * nOutputWeights.size());
            plik.write(result, sizeof(float) * nOutputWeights.size());
        }
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

        // float output = 0;
        // for (int i=0; i < hidden2; i++) {
        //     float score = 0.0;
        //     for (int j=0; j < hidden; j++) {
        //         score += scores[j] * hiddenWeights2[i*hidden+j];
        //     }
        //     score = relu(score);
        //     output += outputWeights[i] * score;
        // }

        vector<float> scores2(hidden2);
        for (int i=0; i < hidden; i++) {
            for (int j=0; j < hidden2; j++) {
                scores2[j] += hiddenWeights2[i*hidden2+j] * scores[i];
            }
        }
        for (int i=0; i < hidden2; i++) {
            scores2[i] = relu(scores2[i]);
        }

        float output = 0;
        for (int i=0; i < hidden2; i++) {
            output += outputWeights[i] * scores2[i];
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

        // float output = 0;
        // for (int i=0; i < hidden2; i++) {
        //     float score = 0.0;
        //     for (int j=0; j < hidden; j++) {
        //         score += scores[j] * hiddenWeights2[i*hidden+j];
        //     }
        //     score = relu(score);
        //     output += outputWeights[i] * score;
        // }

        vector<float> scores2(hidden2);
        for (int i=0; i < hidden; i++) {
            for (int j=0; j < hidden2; j++) {
                scores2[j] += hiddenWeights2[i*hidden2+j] * scores[i];
            }
        }
        for (int i=0; i < hidden2; i++) {
            scores2[i] = relu(scores2[i]);
        }

        float output = 0;
        for (int i=0; i < hidden2; i++) {
            output += outputWeights[i] * scores2[i];
        }

        return fast_tanh(output);
    }

    virtual void learn(const vector<int> & indexes, float target) {
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu2(scores[i]);
        }

        vector<float> scores2(hidden2);
        for (int i=0; i < hidden; i++) {
            for (int j=0; j < hidden2; j++) {
                scores2[j] += hiddenWeights2[i*hidden2+j] * scores[i];
            }
        }
        for (int i=0; i < hidden2; i++) {
            scores2[i] = relu(scores2[i]);
        }

        float op = 0.0;
        for (int i=0; i < hidden2; i++) {
            op += outputWeights[i] * scores2[i];
        }
        op = fast_tanh(op);

        float e = target - op;
        float dop = e * dtanh(op);

        vector<float> eh(hidden2);
        for (int i=0; i < hidden2; i++) {
            eh[i] = dop * outputWeights[i];
        }

        vector<float> dh(hidden2);
        for (int i=0; i < hidden2; i++) {
            dh[i] = eh[i] * drelu(scores2[i]);
        }

        vector<float> dh0(hidden);
        for (int i=0; i < hidden; i++) {
            for (int j=0; j < hidden2; j++) {
                dh0[i] += dh[j] * hiddenWeights2[i*hidden2+j];
            }
            dh0[i] *= drelu2(scores[i]);
        }

        for (int i=0; i < hidden2; i++) {
            outputMomentum[i] = mom * outputMomentum[i] + alpha * scores2[i] * dop;
            outputWeights[i] += outputMomentum[i];
        }

        for (int i=0; i < hidden; i++) {
            for (int j=0; j < hidden2; j++) {
                hiddenMomentum2[i*hidden2+j] = mom * hiddenMomentum2[i*hidden2+j] + alpha * dh[j] * scores[i];
                hiddenWeights2[i*hidden2+j] += hiddenMomentum2[i*hidden2+j];
            }
        }

        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                hiddenMomentum[index*hidden+i] = mom * hiddenMomentum[index*hidden+i] + alpha * dh0[i];
                hiddenWeights[index*hidden+i] += hiddenMomentum[index*hidden+i];
            }
        }
    }

    virtual void learnRL(const vector<int> & indexes, float target) {
        vector<float> scores(hidden);
        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                scores[i] += hiddenWeights[index*hidden+i];
            }
        }
        for (int i=0; i < hidden; i++) {
            scores[i] = relu2(scores[i]);
        }

        vector<float> scores2(hidden2);
        for (int i=0; i < hidden; i++) {
            for (int j=0; j < hidden2; j++) {
                scores2[j] += hiddenWeights2[i*hidden2+j] * scores[i];
            }
        }
        for (int i=0; i < hidden2; i++) {
            scores2[i] = relu(scores2[i]);
        }

        float op = 0.0;
        for (int i=0; i < hidden2; i++) {
            op += outputWeights[i] * scores2[i];
        }
        op = fast_tanh(op);

        float e = target - op;
        float dop = e * dtanh(op);

        vector<float> eh(hidden2);
        for (int i=0; i < hidden2; i++) {
            eh[i] = dop * outputWeights[i];
        }

        vector<float> dh(hidden2);
        for (int i=0; i < hidden2; i++) {
            dh[i] = eh[i] * drelu(scores2[i]);
        }

        vector<float> dh0(hidden);
        for (int i=0; i < hidden; i++) {
            for (int j=0; j < hidden2; j++) {
                dh0[i] += dh[j] * hiddenWeights2[i*hidden2+j];
            }
            dh0[i] *= drelu2(scores[i]);
        }

        for (int i=0; i < hidden2; i++) {
            float a = aOutputWeights[i] == 0 ? 1 : (abs(nOutputWeights[i]) / aOutputWeights[i]);
            outputWeights[i] += 0.2 * a * scores2[i] * dop;
            aOutputWeights[i] += abs(scores2[i] * dop);
            nOutputWeights[i] += scores2[i] * dop;
        }

        for (int i=0; i < hidden; i++) {
            for (int j=0; j < hidden2; j++) {
                float a = aHiddenWeights2[i*hidden2+j] == 0 ? 1 : (abs(nHiddenWeights2[i*hidden2+j]) / aHiddenWeights2[i*hidden2+j]);
                hiddenWeights2[i*hidden2+j] += 0.2 * a * scores[i] * dh[j];
                aHiddenWeights2[i*hidden2+j] += abs(scores[i] * dh[j]);
                nHiddenWeights2[i*hidden2+j] += scores[i] * dh[j];
            }
        }

        for (auto & index : indexes) {
            for (int i=0; i < hidden; i++) {
                float a = aHiddenWeights[index*hidden+i] == 0 ? 1 : (abs(nHiddenWeights[index*hidden+i]) / aHiddenWeights[index*hidden+i]);
                hiddenWeights[index*hidden+i] += 0.2 * a * dh0[i];
                aHiddenWeights[index*hidden+i] += abs(dh0[i]);
                nHiddenWeights[index*hidden+i] += dh0[i];
            }
        }
    }

    virtual void print(float a, float d, string name) {
        float m = 0;
        float sum = 0;
        for (auto & i : hiddenWeights) {
            if (abs(i) > m) m = abs(i);
            int32_t s = (int32_t)round((d*(i+a)));
            float x = (s/d)-a;
            sum += x;
        }
        for (auto & i : hiddenWeights2) {
            if (abs(i) > m) m = abs(i);
            int32_t s = (int32_t)round((d*(i+a)));
            float x = (s/d)-a;
            sum += x;
        }
        for (auto & i : outputWeights) {
            if (abs(i) > m) m = abs(i);
            int32_t s = (int32_t)round((d*(i+a)));
            float x = (s/d)-a;
            sum += x;
        }

        stringstream ss;
        for (auto & i : hiddenWeights) {
            int32_t s = (int32_t)round(d*(i+a));
            char a = (char)(s&255);
            char b = (char)((s>>8)&255);
            ss << b << a;
        }
        for (auto & i : hiddenWeights2) {
            int32_t s = (int32_t)round(d*(i+a));
            char a = (char)(s&255);
            char b = (char)((s>>8)&255);
            ss << b << a;
        }
        for (auto & i : outputWeights) {
            int32_t s = (int32_t)round(d*(i+a));
            char a = (char)(s&255);
            char b = (char)((s>>8)&255);
            ss << b << a;
        }

        cout << "max " << m << endl;
        cout << "sum " << sum << endl;
        ofstream plik; plik.open(name);
        plik << ss.str();
        plik.close();
    }

    float relu2(float x) {
        if (x < 0) return 0.01f * x;
        return x*x;
    }

    float drelu2(float x) {
        if (x < 0) return 0.01f;
        return 2 * sqrtf(x);
    }
};


#endif // NETWORK_H
