#ifndef CPUMCTSTT_H
#define CPUMCTSTT_H

#include <chrono>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "game.h"
#include "network.h"
#include "random.h"
#include "negamaxcpu.h"
#include "mctscpu.h"

using namespace std;
using namespace std::chrono;

class MoveMctsTTR2 {
public:
    int index;
    int parent;
    int player;
    string move;
    bool terminal;

    float heuristic;
    int games;
    float score;
    float virtualLoss;

    int childStart;
    int childrenSize;

    SpinLock* lock;

    explicit MoveMctsTTR2(int parent, int player, const string & move) : parent(parent), player(player), move(move) {
        terminal = false;
        games = 0;
        heuristic = 0;
        score = 0;
        childStart = -1;
        childrenSize = -1;
        index = -1;
        virtualLoss = 0;
    }

    void updateScore(vector<MoveMctsTTR2> & movesPool, float score = 0) {
        lock->lock();
        this->games += 1;
        this->score += score;
        if (childrenSize == -1) {
            virtualLoss -= 1;
            lock->unlock();
            return;
        }
        float h = -INF;
        bool toTerminate = true;

        for (int i=0; i < childrenSize; i++) {
            int index = (childStart+i) & (movesPool.size()-1);
            auto & c = movesPool[index];
            h = max(h,c.heuristic);
            if (c.heuristic > INF/2) {
                toTerminate = true;
            }
            toTerminate &= c.terminal;
        }

        this->heuristic = this->player == movesPool[childStart].player ? h : -h;
        this->terminal = toTerminate || h > INF/2;
        virtualLoss -= 1;
        lock->unlock();
    }

    float updateScore2(vector<MoveMctsTTR2> & movesPool) {
        // lock->lock();
        this->games += 1;
        float h = -INF;
        bool toTerminate = true;

        for (int i=0; i < childrenSize; i++) {
            int index = (childStart+i) & (movesPool.size()-1);
            auto & c = movesPool[index];
            h = max(h,c.heuristic);
            if (c.heuristic > INF/2) {
                toTerminate = true;
            }
            toTerminate &= c.terminal;
        }

        this->heuristic = this->player == movesPool[childStart].player ? h : -h;
        float s = this->heuristic;
        if (s < -1) s = -1;
        if (s > 1) s = 1;
        this->score += s;
        this->terminal = toTerminate || h > INF/2;
        // lock->unlock();
        return s;
    }
};

class CpuMctsTTRWorker0 {
public:
    int id = 0;
    int mainId;
    Network *agent;
    Random ran;
    const int SIZE;
    int moveLimit = 250;

    void shuffle(vector<int> &ints) {
        int i, j, tmp;
        for (i = ints.size() - 1; i > 0; i--) {
            j = ran.nextInt(i + 1);
            tmp = ints[j];
            ints[j] = ints[i];
            ints[i] = tmp;
        }
    }

    uint64_t hashPaths(vector<Path> &paths) {
        uint64_t output = 0;
        for (auto &path : paths) {  // taken from xorshift64
            uint64_t h = 573453117ULL * path.hashCode;
            h ^= h << 13;
            h ^= h >> 7;
            h ^= h << 17;
            output += h;
        }
        return output;
    }

    vector<MoveMctsTTR2> & movesPool;
    mutex & globalLock;
    vector<SpinLock> & spinLocks;
    int ccc = 0;

    float alpha = 0.35f;
    float FPU = 0.5f;
    float C = 0.95f;
    float Croot = 1.0f;
    float VLOSS = 0.5f;

    int & games;
    int & maxLevel;
    bool & provenEnd;

    int jumpTo(int index) {
        return (91153 * index + 5) & (SIZE-1);
    }

    int getMove(int parent, int player, const string & m) {
        int index = (id*SIZE/16) + ccc;
        auto move = &movesPool[index];
        move->lock = &spinLocks[jumpTo(index)];
        move->index = index;
        move->parent = parent;
        move->player = player;
        move->move = m;
        move->heuristic = 0;
        move->games = 0;
        move->score = 0;
        move->virtualLoss = 0;
        move->terminal = false;
        move->childStart = -1;
        move->childrenSize = -1;
        ccc = ((ccc+1)&((SIZE/16)-1));
        return move->index;
    }

    explicit CpuMctsTTRWorker0(int SIZE, vector<MoveMctsTTR2> & movesPool, mutex & globalLock, vector<SpinLock> & spinLocks, int & games, int &maxLevel, bool &provenEnd) :
        SIZE(SIZE), movesPool(movesPool), globalLock(globalLock), spinLocks(spinLocks), games(games), maxLevel(maxLevel), provenEnd(provenEnd) {
    }

    void setPlayer(int player) {
        this->player = player;
    }

    int getPlayer() {
        return player;
    }

    void setGame(Game *game) {
        this->game = game;
    }

    void doWork(high_resolution_clock::time_point start, long timeInMicro, pair<int,int> & childs) {
        long duration = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
        game = new Game(*(this->game));
        Game copy = *game;
        while (!provenEnd && duration < timeInMicro && ccc+moveLimit < (SIZE/16)) {
            globalLock.lock();
            int g = games+1;
            globalLock.unlock();
            selectAndExpand(childs.first, childs.second, g+1,0);
            globalLock.lock();
            games++;
            globalLock.unlock();
            *game = copy;
            duration = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
        }
        delete game;
    }

private:
    int player;
    Game *game;

    vector<MoveMctsTTR2*> getMoves(int childStart, int childrenSize) {
        vector<MoveMctsTTR2*> moves; moves.reserve(childrenSize);
        for (int i=0; i < childrenSize; i++) {
            int index = (childStart+i) & (movesPool.size()-1);
            moves.push_back(&movesPool[index]);
        }
        return moves;
    }

    pair<int,int> generateMoves(int parent) {
        deque<pair<int, vector<Path>>> talia;
        vector<int> vertices; vertices.reserve(25);
        vector<uint64_t> pathCycles; pathCycles.reserve(25);
        vector<int> blockedMoves; blockedMoves.reserve(25);
        string str = "";
        vector<int> ns(8);

        int player = game->currentPlayer;
        int childrenSize = 0;
        int childStart = -1;

        bool alreadyBlocking = true || game->pitch.isCutOffFromOpponentGoal(player == ONE ? TWO : ONE);
        bool alreadyBlocked = true || game->pitch.isCutOffFromOpponentGoal(player);

        vertices.push_back(game->pitch.ball);
        talia.push_back(make_pair(game->pitch.ball, vector<Path>()));


        // vector<int> konceEdges;
        bool check = true;
        if (!game->pitch.onlyOneEmpty()) {
            // konceEdges = game->pitch.fillBlockedKonce();
            check = game->pitch.shouldCheckForGameOver(player);
        }

        int loop = 0;

        vector<int> tsBase = agent->type == 0 ? game->getTuplesEdgesBase() : agent->type == 1 ? game->getTuplesEdgesBase3() : agent->type == 2 ? game->getTuplesEdgesBase4() : game->getTuplesEdgesBase6();
        agent->cacheScore(tsBase,16*mainId+id);

        while (!talia.empty() && childrenSize < moveLimit) {
            pair<int, vector<Path>> v_paths;
            loop++;
            if ((loop & 15) == 0) {
                v_paths = talia.front();
                talia.pop_front();
            } else {
                v_paths = talia.back();
                talia.pop_back();
            }
            int t = v_paths.first;
            vector<Path> &paths = v_paths.second;
            for (auto &p : paths) {
                str += game->pitch.getDistanceChar(p.a, p.b);
                game->pitch.addEdge(p.a, p.b);
                vertices.push_back(p.b);
            }
            game->pitch.ball = t;
            game->pitch.fillFreeNeighbours(ns, t);
            shuffle(ns);

            for (auto & n : ns) {
                if (!game->pitch.isAlmostBlocked(n) && game->pitch.passNext(n)) {
                    vector<Path> newPaths(paths);
                    newPaths.push_back(Path(t, n));
                    if (find(vertices.begin(), vertices.end(), n) != vertices.end()) {
                        int newPathsHash = hashPaths(newPaths);
                        if (find(pathCycles.begin(), pathCycles.end(), newPathsHash) == pathCycles.end()) {
                            pathCycles.push_back(newPathsHash);
                            talia.push_back(make_pair(n, newPaths));
                        }
                    } else
                        talia.push_back(make_pair(n, newPaths));
                } else {
                    player_t goal = game->pitch.goal(n);
                    game->pitch.addEdge(t, n);
                    game->pitch.ball = n;

                    float score = 0;
                    if (goal != NONE) {
                        if (goal == player) {
                            score = MIN_GOAL + game->rounds;
                        } else {
                            score = MAX_GOAL - game->rounds;
                        }
                    } else {
                        if (game->pitch.isBlocked(n)) {
                            vector<Path> newPaths(paths);
                            newPaths.push_back(Path(t, n));
                            int newPathsHash = hashPaths(newPaths);
                            if (find(blockedMoves.begin(), blockedMoves.end(), newPathsHash) != blockedMoves.end()) {
                                game->pitch.ball = t;
                                game->pitch.removeEdge(t, n);
                                continue;
                            }
                            blockedMoves.push_back(newPathsHash);
                            score = MIN_BLOCKED + game->rounds;
                        } else {
                            if (check && game->pitch.isGoalReachable(player)) {
                                score = MIN_GOAL_NEXT_MOVE + game->rounds;
                            } else if (game->pitch.onlyOneEmpty()) {
                                score = MAX_ONE_EMPTY - game->rounds;
                            } else if (game->pitch.onlyTwoEmpty()) {
                                score = MIN_GOAL_ONE_EMPTY + game->rounds;
                            } else if (!alreadyBlocked && game->pitch.passNextDone2(t) && game->pitch.isCutOffFromOpponentGoal(player)) {
                                score = MIN_CUTOFF + game->rounds;
                            } else if (!alreadyBlocking && game->pitch.passNextDone2(t) && game->pitch.isCutOffFromOpponentGoal(player == ONE ? TWO : ONE)) {
                                score = MAX_CUTOFF - game->rounds;
                            } else {
                                game->changePlayer();
                                game->rounds++;
                                // auto tuples = agent->type == 0 ? game->getTuplesEdges() : game->getTuplesEdges3();
                                // float h = agent->getScore(tuples);
                                // score = -h;

                                tsDiff.clear();
                                if (agent->type == 1) {
                                    if (game->currentPlayer == ONE) {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]);
                                        }
                                        tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[105*t+n]);
                                        vector<int> distances(105,9);
                                        game->pitch.calculateDistances(game->pitch.ball,distances);
                                        tsDiff.push_back(316+distances[100]);
                                        tsDiff.push_back(316+10+distances[103]);
                                        tsDiff.push_back(336+n);
                                    } else {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]]);
                                        }
                                        tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[105*t+n]]);
                                        vector<int> distances(105,9);
                                        game->pitch.calculateDistances(game->pitch.ball,distances);
                                        tsDiff.push_back(316+distances[103]);
                                        tsDiff.push_back(316+10+distances[100]);
                                        tsDiff.push_back(336+Pitch::BALLS_MIRRORED[n]);
                                    }
                                } else if (agent->type == 2) {
                                    if (game->currentPlayer == ONE) {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]);
                                        }
                                        tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[105*t+n]);
                                        vector<int> distances(105,9);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        for (int i=0; i < 105; i++) {
                                            tsDiff.push_back(316+10*i+distances[i]);
                                        }
                                        tsDiff.push_back(1366+n);
                                    } else {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]]);
                                        }
                                        tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[105*t+n]]);
                                        vector<int> distances(105,9);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        for (int i=0; i < 105; i++) {
                                            tsDiff.push_back(316+10*i+distances[Pitch::BALLS_MIRRORED[i]]);
                                        }
                                        tsDiff.push_back(1366+Pitch::BALLS_MIRRORED[n]);
                                    }
                                } else if (agent->type == 5) {
                                    if (game->currentPlayer == ONE) {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]);
                                        }
                                        tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[105*t+n]);
                                        vector<int> distances(105,7);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        distances[game->pitch.ball] = 0;
                                        for (int i=0; i < 105; i++) {
                                            int t = distances[i];
                                            t = 9 * (game->pitch.matrixNodes[i] & 0x0F) + t;
                                            tsDiff.push_back(316+80*i+t);
                                        }
                                    } else {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]]);
                                        }
                                        tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[105*t+n]]);
                                        vector<int> distances(105,7);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        distances[game->pitch.ball] = 0;
                                        for (int i=0; i < 105; i++) {
                                            int t = distances[Pitch::BALLS_MIRRORED[i]];
                                            t = 9 * (game->pitch.matrixNodes[Pitch::BALLS_MIRRORED[i]] & 0x0F) + t;
                                            tsDiff.push_back(316+80*i+t);
                                        }
                                    }
                                } else {
                                    if (game->currentPlayer == ONE) {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]);
                                        }
                                        tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[105*t+n]);
                                        vector<int> distances(105,7);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        distances[game->pitch.ball] = 0;
                                        for (int i=0; i < 105; i++) {
                                            int t = distances[i];
                                            if (t == 7) t = 56;
                                            else t = 8 * t + max((game->pitch.matrixNodes[i] & 0x0F) - 1,0);
                                            tsDiff.push_back(316+57*i+t);
                                        }
                                    } else {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]]);
                                        }
                                        tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[105*t+n]]);
                                        vector<int> distances(105,7);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        distances[game->pitch.ball] = 0;
                                        for (int i=0; i < 105; i++) {
                                            int t = distances[Pitch::BALLS_MIRRORED[i]];
                                            if (t == 7) t = 56;
                                            else t = 8 * t + max((game->pitch.matrixNodes[Pitch::BALLS_MIRRORED[i]] & 0x0F) - 1,0);
                                            tsDiff.push_back(316+57*i+t);
                                        }
                                    }
                                }

                                float h2 = agent->getScoreDiff({},tsDiff,16*mainId+id);
                                score = -h2;
                                // if (abs(h-h2) > 0.01f) {
                                //     cout << h << " vs " << h2 << endl;
                                // }

                                game->changePlayer();
                                game->rounds--;
                            }
                        }
                    }
                    game->pitch.ball = t;
                    game->pitch.removeEdge(t, n);
                    str += game->pitch.getDistanceChar(t, n);

                    int childIndex = getMove(parent,player,str);
                    if (childStart == -1) childStart = childIndex;
                    childrenSize++;
                    auto & move = movesPool[childIndex];
                    move.heuristic = score;
                    if (score > 100) {
                        move.heuristic = score;
                        move.score = 1;
                        move.games = 1;
                        move.terminal = true;
                    } else if (score < -100) {
                        move.heuristic = score;
                        move.score = -1;
                        move.games = 1;
                        move.terminal = true;
                    }
                    str.pop_back();
                    if (childrenSize >= moveLimit) break;
                }
            }
            vertices.erase(vertices.begin() + 1, vertices.end());
            str.clear();
            for (auto &p : paths) game->pitch.removeEdge(p.a, p.b);
            game->pitch.ball = vertices[0];
        }

        // for (int i = 0; i < konceEdges.size() / 2; i++) {
        // game->pitch.removeEdge(konceEdges[2 * i], konceEdges[2 * i + 1]);
        // }

        return make_pair(childStart,childrenSize);
    }

    vector<int> tsDiffBase;
    vector<int> tsDiff;

    vector<int> indexes;
    void selectAndExpand(int childStart, int childSize, int games, int level) {
        while (true) {
            maxLevel = max(maxLevel,level);
            indexes.clear();
            float maxArg = -2 * INF;
            float a,b;
            float t = log(games);
            for (int m=0; m < childSize; m++) {
                int i = (childStart+m) & (movesPool.size()-1);
                MoveMctsTTR2 *move = &movesPool[i];
                if (move->terminal) {
                    a = move->heuristic / move->games;
                    a = move->heuristic;
                    if (a == 0.0) a = -1.5;
                    b = 0.0;
                } else {
                    float v = VLOSS * move->virtualLoss;
                    if (move->games == 0) {
                        a = move->heuristic - 0.01f * v;
                        a *= ran.nextFloat(0.95f, 1.05f);
                        b = level == 0 ? 1.0 : FPU;
                    } else {
                        a = alpha * move->heuristic + (1-alpha) * (move->score-v) / (move->games+v);
                        a *= ran.nextFloat(0.9f, 1.1f);
                        b = (level == 0 ? (Croot) : C) * sqrtf(t/(move->games+v));
                    }
                }

                if (a + b > maxArg) {
                    maxArg = a + b;
                    indexes.clear();
                    indexes.push_back(i);
                } else if (a + b == maxArg) {
                    indexes.push_back(i);
                }
            }
            MoveMctsTTR2 *move = &movesPool[indexes[ran.nextInt(indexes.size())]];
            move->lock->lock();
            move->virtualLoss += 1;
            if (move->terminal) {
                move->games += 1;
                move->score += move->heuristic > INF/2 ? 1 : move->heuristic < -INF/2 ? -1 : 0;
                move->virtualLoss -= 1;
                move->lock->unlock();
                if (level == 0) {
                    if (move->heuristic > INF/2) {
                        provenEnd = true;
                        return;
                    }
                    bool allTerminal = true;
                    for (int m=0; m < childSize; m++) {
                        int i = (childStart+m) & (movesPool.size()-1);
                        MoveMctsTTR2 *move = &movesPool[i];
                        if (!move->terminal) {
                            allTerminal = false;
                            break;
                        }
                    }
                    if (allTerminal) {
                        provenEnd = true;
                        return;
                    }
                }
                MoveMctsTTR2 *parent = move->parent == -1 ? nullptr : &movesPool[move->parent];
                while (parent != nullptr) {
                    parent->updateScore(movesPool, parent->player == move->player ? (move->heuristic > INF/2 ? 1 : move->heuristic < -INF/2 ? -1 : 0) : (move->heuristic > INF/2 ? -1 : move->heuristic < -INF/2 ? 1 : 0));
                    parent = parent->parent == -1 ? nullptr : &movesPool[parent->parent];
                }
                return;
            }
            game->makeMove(move->move);
            // if (move->games > 0) {
            //     if (move->childrenSize == -1) {
            //         auto children = generateMoves(move->index);
            //         move->childStart = children.first;
            //         move->childrenSize = children.second;
            //     }
            //     move->lock->unlock();
            //     childStart = move->childStart;
            //     childSize = move->childrenSize;
            //     level = level+1;
            //     games = move->games+1;
            // } else {
            //     float s = move->heuristic;
            //     move->games += 1;
            //     move->score += s;
            //     move->virtualLoss -= 1;
            //     move->lock->unlock();
            //     MoveMctsTTR2 *parent = move->parent == -1 ? nullptr : &movesPool[move->parent];
            //     while (parent != nullptr) {
            //         parent->updateScore(movesPool, parent->player == move->player ? s : -s);
            //         parent = parent->parent == -1 ? nullptr : &movesPool[parent->parent];
            //     }
            //     return;
            // }
            if (move->games > 0) {
                if (move->childrenSize == -1) {
                    auto children = generateMoves(move->index);
                    move->childStart = children.first;
                    move->childrenSize = children.second;
                }
                move->lock->unlock();
                childStart = move->childStart;
                childSize = move->childrenSize;
                level = level+1;
                games = move->games+1;
            } else {
                auto children = generateMoves(move->index);
                move->childStart = children.first;
                move->childrenSize = children.second;
                float score = move->updateScore2(movesPool);
                move->virtualLoss -= 1;
                move->lock->unlock();
                MoveMctsTTR2 *parent = move->parent == -1 ? nullptr : &movesPool[move->parent];
                while (parent != nullptr) {
                    parent->updateScore(movesPool, parent->player == move->player ? score : -score);
                    parent = parent->parent == -1 ? nullptr : &movesPool[parent->parent];
                }
                return;
            }
        }
    }
};

class CpuMctsTTRParallel0 {
public:
    int id = 0;
    Network *agent;
    Random ran;
    const int SIZE;
    int moveLimit = 250;

    void shuffle(vector<int> &ints) {
        int i, j, tmp;
        for (i = ints.size() - 1; i > 0; i--) {
            j = ran.nextInt(i + 1);
            tmp = ints[j];
            ints[j] = ints[i];
            ints[i] = tmp;
        }
    }

    uint64_t hashPaths(vector<Path> &paths) {
        uint64_t output = 0;
        for (auto &path : paths) {  // taken from xorshift64
            uint64_t h = 573453117ULL * path.hashCode;
            h ^= h << 13;
            h ^= h >> 7;
            h ^= h << 17;
            output += h;
        }
        return output;
    }

    vector<MoveMctsTTR2> movesPool = vector<MoveMctsTTR2>(SIZE,MoveMctsTTR2(-1,-1,""));
    vector<SpinLock> spinLocks = vector<SpinLock>(SIZE);
    mutex globalLock;
    vector<CpuMctsTTRWorker0*> workers;
    int ccc = 0;

    float alpha = 0.35f;
    float FPU = 0.5f;
    float C = 0.95f;
    float Croot = 1.0f;
    float VLOSS = 0.5f;

    int games;
    int maxLevel;
    bool provenEnd;

    int jumpTo(int index) {
        return (91153 * index + 5) & (SIZE-1);
    }

    int getMove(int parent, int player, const string & m) {
        int index = (0*SIZE/16) + ccc;
        auto move = &movesPool[index];
        move->lock = &spinLocks[jumpTo(index)];
        move->index = index;
        move->parent = parent;
        move->player = player;
        move->move = m;
        move->heuristic = 0;
        move->games = 0;
        move->score = 0;
        move->virtualLoss = 0;
        move->terminal = false;
        move->childStart = -1;
        move->childrenSize = -1;
        ccc = ((ccc+1)&((SIZE/16)-1));
        return move->index;
    }

    explicit CpuMctsTTRParallel0(int SIZE = 4194304) : SIZE(SIZE) {
        workers.reserve(16);
        for (int i=0; i < 16; i++) {
            CpuMctsTTRWorker0* worker = new CpuMctsTTRWorker0(SIZE,movesPool,globalLock,spinLocks,games,maxLevel,provenEnd);
            workers.push_back(worker);
        }
    }

    virtual ~CpuMctsTTRParallel0() {
        for (auto & w : workers) delete w;
    }

    void setPlayer(int player) {
        this->player = player;
    }

    int getPlayer() {
        return player;
    }

    void setGame(Game *game) {
        this->game = game;
    }

    void doWork(chrono::high_resolution_clock::time_point start, long timeInMicro, pair<int,int> & childs, int id) {
        auto & worker = workers[id];
        worker->agent = agent;
        worker->setGame(game);
        worker->setPlayer(player);
        worker->mainId = this->id;
        worker->id = id;
        worker->FPU = FPU;
        worker->alpha = alpha;
        worker->C = C;
        worker->Croot = Croot;
        worker->VLOSS = VLOSS;
        worker->moveLimit = moveLimit;
        if (id == 0) worker->ccc = ccc;
        else worker->ccc = 0;
        worker->doWork(start,timeInMicro,childs);
    }

    stringstream ss;

    MoveMctsTTR2* getBestMove(long timeInMicro, int th, bool print = false) {
        auto start = high_resolution_clock::now();
        vector<MoveMctsTTR2*> moves;
        pair<int,int> childs;

        games = 0;
        maxLevel = 0;
        provenEnd = false;
        ccc = 0;
        ss = stringstream();

        if (moves.size() == 0) {
            childs = generateMoves(-1);
            moves = getMoves(childs.first, childs.second);
        }

        vector<thread> threads; threads.reserve(th);
        for (int i=0; i < th; i++) {
            threads.push_back(thread(&CpuMctsTTRParallel0::doWork, this, start, timeInMicro, ref(childs), i));
        }
        for (auto & t : threads) {
            t.join();
        }

        sort(moves.begin(),moves.end(), [](const MoveMctsTTR2 *a, const MoveMctsTTR2 *b) -> bool
             {
                 float avg1 = a->heuristic + log(a->games+3);
                 float avg2 = b->heuristic + log(b->games+3);
                 return avg1 > avg2;
             });

        if (print)
            for (auto & m : moves) {
                cout << m->move << ": " << m->score/m->games << " " << m->heuristic << " " << m->games << endl;
                break;
            }

        // cout << "games: " << games << endl;
        //                cout << "maxLevel: " << maxLevel << endl;

        for (auto & m : moves) {
            float h = 0.5f + (m->score / (m->games==0?1:m->games)) / 2.0f;
            if (m->games == 0) h = 0.5f + (m->heuristic / (m->games==0?1:m->games)) / 2.0f;
            if (m->terminal) h = m->heuristic > 100 ? 1 : 0;
            h = round(10000 * h) / 100.0f;
            ss << m->move << ": " << h << "%" << endl;
        }
        // cout << "games: " << games << endl;
        // cout << "maxLevel: " << maxLevel << endl;
        ss << "-------------------------------------------------" << endl;
        ss << "possible moves: " << moves.size() << endl;
        ss << "visits: " << games << endl;
        // ss << "expansions: " << expansions << endl;
        int cccSum = 0;
        for (int i=0; i < th; i++) cccSum += workers[i]->ccc;
        ss << "nodes: " << cccSum << endl;
        ss << "maxLevel: " << maxLevel << endl;
        float h = 0.5f + (moves[0]->score / (moves[0]->games==0?1:moves[0]->games)) / 2.0f;
        if (moves[0]->games == 0) h = 0.5f + (moves[0]->heuristic / (moves[0]->games==0?1:moves[0]->games)) / 2.0f;
        if (moves[0]->terminal) h = moves[0]->heuristic > 100 ? 1 : 0;
        h = round(10000 * h) / 100.0f;
        ss << "best move: " << moves[0]->move << ": " << h << "%" << endl;
        ss << "-------------------------------------------------" << endl;

        return moves[0];
    }

private:
    int player;
    Game *game;

    vector<MoveMctsTTR2*> getMoves(int childStart, int childrenSize) {
        vector<MoveMctsTTR2*> moves; moves.reserve(childrenSize);
        for (int i=0; i < childrenSize; i++) {
            int index = (childStart+i) & (movesPool.size()-1);
            moves.push_back(&movesPool[index]);
        }
        return moves;
    }

    pair<int,int> generateMoves(int parent) {
        deque<pair<int, vector<Path>>> talia;
        vector<int> vertices; vertices.reserve(25);
        vector<uint64_t> pathCycles; pathCycles.reserve(25);
        vector<int> blockedMoves; blockedMoves.reserve(25);
        string str = "";
        vector<int> ns(8);

        int player = game->currentPlayer;
        int childrenSize = 0;
        int childStart = -1;

        bool alreadyBlocking = true || game->pitch.isCutOffFromOpponentGoal(player == ONE ? TWO : ONE);
        bool alreadyBlocked = true || game->pitch.isCutOffFromOpponentGoal(player);

        vertices.push_back(game->pitch.ball);
        talia.push_back(make_pair(game->pitch.ball, vector<Path>()));


        vector<int> konceEdges;
        bool check = true;
        if (!game->pitch.onlyOneEmpty()) {
            konceEdges = game->pitch.fillBlockedKonce();
            check = game->pitch.shouldCheckForGameOver(player);
        }

        int loop = 0;

        vector<int> tsBase = agent->type == 0 ? game->getTuplesEdgesBase() : agent->type == 1 ? game->getTuplesEdgesBase3() : agent->type == 2 ? game->getTuplesEdgesBase4() : game->getTuplesEdgesBase6();
        agent->cacheScore(tsBase,16*id);

        while (!talia.empty() && childrenSize < moveLimit) {
            pair<int, vector<Path>> v_paths;
            loop++;
            if ((loop & 15) == 0) {
                v_paths = talia.front();
                talia.pop_front();
            } else {
                v_paths = talia.back();
                talia.pop_back();
            }
            int t = v_paths.first;
            vector<Path> &paths = v_paths.second;
            for (auto &p : paths) {
                str += game->pitch.getDistanceChar(p.a, p.b);
                game->pitch.addEdge(p.a, p.b);
                vertices.push_back(p.b);
            }
            game->pitch.ball = t;
            game->pitch.fillFreeNeighbours(ns, t);
            shuffle(ns);

            for (auto & n : ns) {
                if (!game->pitch.isAlmostBlocked(n) && game->pitch.passNext(n)) {
                    vector<Path> newPaths(paths);
                    newPaths.push_back(Path(t, n));
                    if (find(vertices.begin(), vertices.end(), n) != vertices.end()) {
                        int newPathsHash = hashPaths(newPaths);
                        if (find(pathCycles.begin(), pathCycles.end(), newPathsHash) == pathCycles.end()) {
                            pathCycles.push_back(newPathsHash);
                            talia.push_back(make_pair(n, newPaths));
                        }
                    } else
                        talia.push_back(make_pair(n, newPaths));
                } else {
                    player_t goal = game->pitch.goal(n);
                    game->pitch.addEdge(t, n);
                    game->pitch.ball = n;

                    float score = 0;
                    if (goal != NONE) {
                        if (goal == player) {
                            score = MIN_GOAL + game->rounds;
                        } else {
                            score = MAX_GOAL - game->rounds;
                        }
                    } else {
                        if (game->pitch.isBlocked(n)) {
                            vector<Path> newPaths(paths);
                            newPaths.push_back(Path(t, n));
                            int newPathsHash = hashPaths(newPaths);
                            if (find(blockedMoves.begin(), blockedMoves.end(), newPathsHash) != blockedMoves.end()) {
                                game->pitch.ball = t;
                                game->pitch.removeEdge(t, n);
                                continue;
                            }
                            blockedMoves.push_back(newPathsHash);
                            score = MIN_BLOCKED + game->rounds;
                        } else {
                            if (check && game->pitch.isGoalReachable(player)) {
                                score = MIN_GOAL_NEXT_MOVE + game->rounds;
                            } else if (game->pitch.onlyOneEmpty()) {
                                score = MAX_ONE_EMPTY - game->rounds;
                            } else if (game->pitch.onlyTwoEmpty()) {
                                score = MIN_GOAL_ONE_EMPTY + game->rounds;
                            } else if (!alreadyBlocked && game->pitch.passNextDone2(t) && game->pitch.isCutOffFromOpponentGoal(player)) {
                                score = MIN_CUTOFF + game->rounds;
                            } else if (!alreadyBlocking && game->pitch.passNextDone2(t) && game->pitch.isCutOffFromOpponentGoal(player == ONE ? TWO : ONE)) {
                                score = MAX_CUTOFF - game->rounds;
                            } else {
                                game->changePlayer();
                                game->rounds++;
                                // auto tuples = agent->type == 0 ? game->getTuplesEdges() : game->getTuplesEdges3();
                                // float h = agent->getScore(tuples);
                                // score = -h;

                                tsDiff.clear();
                                if (agent->type == 1) {
                                    if (game->currentPlayer == ONE) {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]);
                                        }
                                        tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[105*t+n]);
                                        vector<int> distances(105,9);
                                        game->pitch.calculateDistances(game->pitch.ball,distances);
                                        tsDiff.push_back(316+distances[100]);
                                        tsDiff.push_back(316+10+distances[103]);
                                        tsDiff.push_back(336+n);
                                    } else {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]]);
                                        }
                                        tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[105*t+n]]);
                                        vector<int> distances(105,9);
                                        game->pitch.calculateDistances(game->pitch.ball,distances);
                                        tsDiff.push_back(316+distances[103]);
                                        tsDiff.push_back(316+10+distances[100]);
                                        tsDiff.push_back(336+Pitch::BALLS_MIRRORED[n]);
                                    }
                                } else if (agent->type == 2) {
                                    if (game->currentPlayer == ONE) {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]);
                                        }
                                        tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[105*t+n]);
                                        vector<int> distances(105,9);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        for (int i=0; i < 105; i++) {
                                            tsDiff.push_back(316+10*i+distances[i]);
                                        }
                                        tsDiff.push_back(1366+n);
                                    } else {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]]);
                                        }
                                        tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[105*t+n]]);
                                        vector<int> distances(105,9);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        for (int i=0; i < 105; i++) {
                                            tsDiff.push_back(316+10*i+distances[Pitch::BALLS_MIRRORED[i]]);
                                        }
                                        tsDiff.push_back(1366+Pitch::BALLS_MIRRORED[n]);
                                    }
                                } else if (agent->type == 5) {
                                    if (game->currentPlayer == ONE) {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]);
                                        }
                                        tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[105*t+n]);
                                        vector<int> distances(105,7);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        distances[game->pitch.ball] = 0;
                                        for (int i=0; i < 105; i++) {
                                            int t = distances[i];
                                            t = 9 * (game->pitch.matrixNodes[i] & 0x0F) + t;
                                            tsDiff.push_back(316+80*i+t);
                                        }
                                    } else {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]]);
                                        }
                                        tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[105*t+n]]);
                                        vector<int> distances(105,7);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        distances[game->pitch.ball] = 0;
                                        for (int i=0; i < 105; i++) {
                                            int t = distances[Pitch::BALLS_MIRRORED[i]];
                                            t = 9 * (game->pitch.matrixNodes[Pitch::BALLS_MIRRORED[i]] & 0x0F) + t;
                                            tsDiff.push_back(316+80*i+t);
                                        }
                                    }
                                } else {
                                    if (game->currentPlayer == ONE) {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]);
                                        }
                                        tsDiff.push_back(Pitch::ALL_EDGES_INDEXES[105*t+n]);
                                        vector<int> distances(105,7);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        distances[game->pitch.ball] = 0;
                                        for (int i=0; i < 105; i++) {
                                            int t = distances[i];
                                            if (t == 7) t = 56;
                                            else t = 8 * t + max((game->pitch.matrixNodes[i] & 0x0F) - 1,0);
                                            tsDiff.push_back(316+57*i+t);
                                        }
                                    } else {
                                        for (auto & p : paths) {
                                            tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[p.a*105+p.b]]);
                                        }
                                        tsDiff.push_back(Pitch::EDGES_MIRRORED[Pitch::ALL_EDGES_INDEXES[105*t+n]]);
                                        vector<int> distances(105,7);
                                        game->pitch.calculateDistances2(game->pitch.ball,distances);
                                        distances[game->pitch.ball] = 0;
                                        for (int i=0; i < 105; i++) {
                                            int t = distances[Pitch::BALLS_MIRRORED[i]];
                                            if (t == 7) t = 56;
                                            else t = 8 * t + max((game->pitch.matrixNodes[Pitch::BALLS_MIRRORED[i]] & 0x0F) - 1,0);
                                            tsDiff.push_back(316+57*i+t);
                                        }
                                    }
                                }

                                float h2 = agent->getScoreDiff({},tsDiff,16*id);
                                score = -h2;
                                // if (abs(h-h2) > 0.01f) {
                                //     cout << h << " vs " << h2 << endl;
                                // }

                                game->changePlayer();
                                game->rounds--;
                            }
                        }
                    }
                    game->pitch.ball = t;
                    game->pitch.removeEdge(t, n);
                    str += game->pitch.getDistanceChar(t, n);

                    int childIndex = getMove(parent,player,str);
                    if (childStart == -1) childStart = childIndex;
                    childrenSize++;
                    auto & move = movesPool[childIndex];
                    move.heuristic = score;
                    if (score > 100) {
                        move.heuristic = score;
                        move.score = 1;
                        move.games = 1;
                        move.terminal = true;
                    } else if (score < -100) {
                        move.heuristic = score;
                        move.score = -1;
                        move.games = 1;
                        move.terminal = true;
                    }
                    str.pop_back();
                    if (childrenSize >= moveLimit) break;
                }
            }
            vertices.erase(vertices.begin() + 1, vertices.end());
            str.clear();
            for (auto &p : paths) game->pitch.removeEdge(p.a, p.b);
            game->pitch.ball = vertices[0];
        }

        for (int i = 0; i < konceEdges.size() / 2; i++) {
            game->pitch.removeEdge(konceEdges[2 * i], konceEdges[2 * i + 1]);
        }

        return make_pair(childStart,childrenSize);
    }

    vector<int> tsDiffBase;
    vector<int> tsDiff;
};

#endif // CPUMCTSTT_H
