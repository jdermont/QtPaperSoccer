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


#endif // NETWORK_H
