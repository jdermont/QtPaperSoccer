#ifndef MCTSCPU_H
#define MCTSCPU_H

#define LIMIT_MOVES_MCTS 200
#define LIMIT_MOVES_PLAYOUT 30
#define VIRTUAL_LOSS 2
#define EXP_C 0.65
// 1.4142
#include <algorithm>
#include <chrono>
#include <vector>
#include <deque>
#include <atomic>
#include <memory>
#include <thread>
#include <mutex>

#include "cpu.h"
#include "negamaxcpu.h"
#include "pitch.h"
#include "game.h"
#include "random.h"

using namespace std;
using namespace std::chrono;

class SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT ;
public:
    void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) { ; }
    }
    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

class ShortMove {
public:
    float score;
    string move;

    explicit ShortMove(float score = 0, string move = "") : score(score), move(move) {

    }

};

class MctsMove : public IMove {
public:
    player_t player;
    float score;
    int games;
    bool terminal;
    int heuristic;

    MctsMove *parent;
    vector<MctsMove*> children;

    shared_ptr<SpinLock> lock;
    int virtualLoss;

    MctsMove(string move, player_t player, MctsMove *parent) : IMove(move), player(player), parent(parent) {
        score = 0;
        games = 0;
        terminal = false;
        heuristic = 0;
        virtualLoss = 0;
        lock.reset(new SpinLock());
    }

    virtual ~MctsMove() { for (auto & m : children) { delete m; } }

    void updateScore(int score, bool isTerminal) {
        lock->lock();
        if (!terminal) {
            if (isTerminal) {
                this->score = score;
                terminal = true;
            } else {
                this->score += score;
            }
            games++;
        }
        virtualLoss -= VIRTUAL_LOSS;
        lock->unlock();
    }

    bool operator < (const MctsMove& move) const {
        float a = (float) score / (games+1e-3);
        float b = (float) move.score / (move.games+1e-3);
        return a < b;
    }

    bool operator > (const MctsMove& move) const {
        float a = (float) score / (games+1e-3);
        float b = (float) move.score / (move.games+1e-3);
        return a > b;
    }
};

class MctsCpuWorker : public ICpu {
public:
    bool kupa = false;
    MctsCpuWorker(int limit = LIMIT_MOVES, int limitPlayouts = LIMIT_MOVES_PLAYOUT) : ICpu(), limit(limit), limitPlayouts(limitPlayouts) {

    }
    //protected:
    int limit;
    int limitPlayouts;
    bool alreadyBlocking;
    bool alreadyBlocked;
    bool provenEnd;

    vector<MctsMove*> generateMoves(MctsMove* parent, player_t player) {
        vector<MctsMove*> moves; moves.reserve(50);
        deque<pair<int, vector<Path>>> talia;
        vector<int> vertices; vertices.reserve(25);
        vector<uint64_t> pathCycles; pathCycles.reserve(25);
        vector<int> blockedMoves; blockedMoves.reserve(25);
        string str = "";
        vector<int> ns(8);

        vertices.push_back(game->pitch.ball);
        talia.push_back(make_pair(game->pitch.ball, vector<Path>()));

        vector<int> konceEdges;
        bool check = true;
        if (!game->pitch.onlyOneEmpty()) {
            konceEdges = game->pitch.fillBlockedKonce();
            check = game->pitch.shouldCheckForGameOver(player);
        }

        bool alreadyBlocking = player == this->player ? this->alreadyBlocking : this->alreadyBlocked;
        bool alreadyBlocked = player == this->player ? this->alreadyBlocked : this->alreadyBlocking;

        int loop = 0;

        while (!talia.empty() && moves.size() < limit) {
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

                    int score = 0;
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
                                if (true) { // kupa
                                    score = -getScoreQuick(player == ONE ? TWO : ONE, LIMIT_MOVES_MCTS);
                                    if (score == 0) {
                                        score = game->pitch.getDistancesToGoal(player);
                                    }
                                } else {
                                    score = game->pitch.getDistancesToGoal(player);
                                }
                            }
                        }
                    }
                    game->pitch.ball = t;
                    game->pitch.removeEdge(t, n);
                    str += game->pitch.getDistanceChar(t, n);
                    MctsMove *move = new MctsMove(str,player,parent);
                    if (score > TERMINAL_THRESHOLD) {
                        move->score = INF + score;
                        move->games = 1;
                        move->terminal = true;
                    } else if (score < -TERMINAL_THRESHOLD) {
                        move->score = -INF + score;
                        move->games = 1;
                        move->terminal = true;
                    } else {
                        move->heuristic = score;
                    }
                    moves.push_back(move);
                    str.pop_back();
                    if (moves.size() >= limit) break;
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

        return moves;
    }

    void selectAndExpand(vector<MctsMove*> moves, int games, int level) {
        vector<int> indexes;
        float maxArg = -2 * INF;
        float a,b;
        float t = log(games);
        for (size_t i=0; i < moves.size(); i++) {
            MctsMove *move = moves[i];
            int g = move->games + move->virtualLoss;
            if (move->terminal) {
                a = move->score / g;
                b = 0.0;
            } else if (g == 0) {
                a = 1000.0;
                b = 1000.0;

                a = 0.5;
                b = 0.5;

                a += move->heuristic / 10.0;
            } else {
                a = move->score / g;
                a += (move->heuristic-1.0)/(g+1.0);
                b = EXP_C * sqrtf(t / g);
            }

            if (a + b > maxArg) {
                maxArg = a + b;
                indexes.clear();
                indexes.push_back(i);
            } else if (a + b == maxArg) {
                indexes.push_back(i);
            }
        }

        MctsMove *move = moves[indexes[ran.nextInt(indexes.size())]];
        move->lock->lock();
        move->virtualLoss += VIRTUAL_LOSS;
        if (move->terminal) {
            move->virtualLoss -= VIRTUAL_LOSS;
            move->lock->unlock();
            if (move->score > INF/2) {
                if (level == 0) {
                    provenEnd = true;
                    return;
                }
                MctsMove *parent = move->parent;
                if (parent != nullptr) {
                    parent->updateScore(-INF, true);
                    parent = parent->parent;
                }
                while (parent != nullptr) {
                    parent->updateScore(parent->player == move->player ? 1 : 0, false);
                    parent = parent->parent;
                }
            } else {
                bool allChildrenBad = true;
                for (auto &m : moves) {
                    if (m->score < -INF/2) {
                    } else {
                        allChildrenBad = false;
                        break;
                    }
                }
                MctsMove *parent = move->parent;
                if (allChildrenBad) {
                    if (level == 0) {
                        provenEnd = true;
                        return;
                    }
                    if (parent != nullptr) {
                        parent->updateScore(INF, true);
                        parent = parent->parent;

                        if (parent != nullptr) {
                            parent->updateScore(-INF, true);
                            parent = parent->parent;
                        }
                    }
                    while (parent != nullptr) {
                        parent->updateScore(parent->player == move->player ? 0 : 1, false);
                        parent = parent->parent;
                    }
                } else {
                    while (parent != nullptr) {
                        parent->updateScore(parent->player == move->player ? 0 : 1, false);
                        parent = parent->parent;
                    }
                }
            }
            return;
        }
        vector<Path> paths = makeMove(move->move);
        game->rounds++;
        vector<int> konceEdges = game->pitch.fillBlockedKonce();
        if (move->games > 0) {
            if (move->children.empty()) {
                move->children = generateMoves(move, move->player==ONE?TWO:ONE);
            }
            move->lock->unlock();
            selectAndExpand(move->children,move->games+1,level+1);
        } else {
            float score = simulateOne(move->player);
            move->score += score;
            move->games++;
            move->virtualLoss -= VIRTUAL_LOSS;
            move->lock->unlock();

            MctsMove *parent = move->parent;
            while (parent != nullptr) {
                parent->updateScore(parent->player == move->player ? score : 1-score, false);
                parent = parent->parent;
            }
        }
        undoMove(paths);
        game->rounds--;
        for (int i = 0; i < konceEdges.size() / 2; i++) {
            game->pitch.removeEdge(konceEdges[2 * i], konceEdges[2 * i + 1]);
        }
    }

    float simulateOne(int forPlayer) {
        player_t currentPlayer = forPlayer == ONE ? TWO : ONE;
        vector<vector<Path>> paths;
        while (true) {
            vector<ShortMove *> moves = getMovesForSimulation(currentPlayer, limitPlayouts);
            sort(moves.begin(), moves.end(), [](ShortMove *a, ShortMove *b) -> bool { return a->score > b->score; });
            if (abs(moves[0]->score) > TERMINAL_THRESHOLD) {
                int score = moves[0]->score;
                while (!paths.empty()) {
                    vector<Path> p = paths.back();
                    paths.pop_back();
                    undoMove(p);
                }
                for (auto m : moves) {
                    delete m;
                }
                return score > TERMINAL_THRESHOLD ? (currentPlayer == forPlayer ? 1 : 0) : (currentPlayer == forPlayer ? 0 : 1);
            }

            int n = 0;
            for (auto & m : moves) {
                if (m->score < -TERMINAL_THRESHOLD) break;
                n++;
            }

            auto & move = moves[ran.nextInt(n)];

            vector<Path> p = makeMove(move->move);
            paths.push_back(p);

            for (auto m : moves) {
                delete m;
            }

            currentPlayer = currentPlayer == ONE ? TWO : ONE;
        }

        return 0;
    }

    vector<ShortMove *> getMovesForSimulation(player_t player, int limit) {
        vector<ShortMove*> moves; moves.reserve(limit);
        deque<pair<int, vector<Path>>> talia;
        vector<int> vertices; vertices.reserve(25);
        vector<uint64_t> pathCycles; pathCycles.reserve(25);
        vector<int> blockedMoves; blockedMoves.reserve(25);
        string str = "";
        vector<int> ns(8);

        vertices.push_back(game->pitch.ball);
        talia.push_back(make_pair(game->pitch.ball, vector<Path>()));

        vector<int> konceEdges;
        bool check = true;
        if (!game->pitch.onlyOneEmpty()) {
            konceEdges = game->pitch.fillBlockedKonce();
            check = game->pitch.shouldCheckForGameOver(player);
        }

        bool alreadyBlocking = player == this->player ? this->alreadyBlocking : this->alreadyBlocked;
        bool alreadyBlocked = player == this->player ? this->alreadyBlocked : this->alreadyBlocking;

        int loop = 0;

        while (!talia.empty() && moves.size() < limit) {
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
                                if (kupa) {
                                    score = -getScoreQuick(player == ONE ? TWO : ONE, LIMIT_MOVES_PLAYOUT);
                                }
                            }
                        }
                    }
                    game->pitch.ball = t;
                    game->pitch.removeEdge(t, n);
                    str += game->pitch.getDistanceChar(t, n);
                    ShortMove *move = new ShortMove(score,str);
                    moves.push_back(move);
                    if (score > TERMINAL_THRESHOLD) {
                        for (int i = 0; i < konceEdges.size() / 2; i++) {
                            game->pitch.removeEdge(konceEdges[2 * i], konceEdges[2 * i + 1]);
                        }
                        for (auto &p : paths) game->pitch.removeEdge(p.a, p.b);
                        game->pitch.ball = vertices[0];
                        return moves;
                    }
                    str.pop_back();
                    if (moves.size() >= limit) break;
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

        return moves;
    }

    vector<Path> makeMove(string &move) {
        vector<Path> paths;
        paths.reserve(move.size());
        for (auto &c : move) {
            int n = nextNode(c);
            paths.push_back(Path(game->pitch.ball, n));
            game->pitch.addEdge(game->pitch.ball, n);
            game->pitch.ball = n;
        }
        return paths;
    }

    void undoMove(vector<Path> &paths) {
        game->pitch.ball = paths[0].a;
        for (auto &path : paths) {
            game->pitch.removeEdge(path.a, path.b);
        }
    }

    float getScoreQuick(player_t player, int limit) {
        deque<pair<int, vector<Path>>> talia;
        vector<int> vertices; vertices.reserve(25);
        vector<uint64_t> pathCycles; pathCycles.reserve(25);
        vector<int> blockedMoves; blockedMoves.reserve(25);
        vector<int> ns(8);

        vertices.push_back(game->pitch.ball);
        talia.push_back(make_pair(game->pitch.ball, vector<Path>()));

        vector<int> konceEdges;
        bool check = true;
        if (!game->pitch.onlyOneEmpty()) {
            konceEdges = game->pitch.fillBlockedKonce();
            check = game->pitch.shouldCheckForGameOver(player);
        }

        bool alreadyBlocking = player == this->player ? this->alreadyBlocking : this->alreadyBlocked;
        bool alreadyBlocked = player == this->player ? this->alreadyBlocked : this->alreadyBlocking;

        int loop = 0;
        int count = 0;
        int output = -INF;
        while (!talia.empty() && count < limit) {
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

                    int score = 0;
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
                            }
                        }
                    }
                    game->pitch.ball = t;
                    game->pitch.removeEdge(t, n);
                    output = max(score, output);
                    if (output > -TERMINAL_THRESHOLD) {
                        for (int i = 0; i < konceEdges.size() / 2; i++) {
                            game->pitch.removeEdge(konceEdges[2 * i], konceEdges[2 * i + 1]);
                        }
                        for (auto &p : paths) game->pitch.removeEdge(p.a, p.b);
                        game->pitch.ball = vertices[0];
                        return output;
                    }
                    count++;
                    if (count >= limit) break;
                }
            }
            vertices.erase(vertices.begin() + 1, vertices.end());
            for (auto &p : paths) game->pitch.removeEdge(p.a, p.b);
            game->pitch.ball = vertices[0];
        }

        for (int i = 0; i < konceEdges.size() / 2; i++) {
            game->pitch.removeEdge(konceEdges[2 * i], konceEdges[2 * i + 1]);
        }

        if (output == -INF) {
            output = MIN_GOAL_ONE_EMPTY - game->rounds;
        }

        return output;
    }

};

class MctsCpu : public MctsCpuWorker {
public:
    MctsCpu(int limit = LIMIT_MOVES, int limitPlayouts = LIMIT_MOVES_PLAYOUT) : MctsCpuWorker(limit,limitPlayouts) {
        numThreads = 1;
    }

    void work(vector<MctsMove*> &moves, mutex &globalLock) {
        MctsCpuWorker cpuWorker(limit, limitPlayouts);
        cpuWorker.provenEnd = false;
        cpuWorker.kupa = kupa;
        Game *copy = new Game();
        copy->setGame(game);
        cpuWorker.setGame(copy);

        auto duration = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
        while (duration < timeInMicro && !cpuWorker.provenEnd) {
            cpuWorker.selectAndExpand(moves, games + 1, 0);
            globalLock.lock();
            games++;
            globalLock.unlock();
            duration = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
        }

        cpuWorker.deleteGame();
    }

    MctsMove* getBestMove(uint64_t timeInMicro) {
        start = high_resolution_clock::now();
        this->timeInMicro = timeInMicro;
        alreadyBlocking = game->pitch.isCutOffFromOpponentGoal(player == ONE ? TWO : ONE);
        alreadyBlocked = game->pitch.isCutOffFromOpponentGoal(player);

        vector<MctsMove*> moves = generateMoves(nullptr,this->player);

        games = 0;
        provenEnd = false;
        vector<thread> threads;
        mutex globalLock;
        for (int i = 0; i < numThreads; i++) {
            threads.push_back(thread(&MctsCpu::work, this, ref(moves), ref(globalLock)));
        }
        for (auto &t : threads) t.join();

        //        provenEnd = false;
        //        auto duration = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
        //        while (!provenEnd && duration < timeInMicro) {
        //            selectAndExpand(moves, games+1, 0);
        //            games++;
        //            duration = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
        //        }

        //        cout << games << " " << (kupa?"kupa":"") << endl;

        sort(moves.rbegin(), moves.rend(), [](const MctsMove *a, const MctsMove *b) -> bool {
            float A = (float) a->score / (a->games+1e-3);
            float B = (float) b->score / (b->games+1e-3);
            return A < B;
        });

        for (int i=1; i < moves.size(); i++) {
            delete moves[i];
        }

        return moves[0];
    }

    void setNumThreads(int n) {
        numThreads = n;
    }

private:
    int numThreads;
    int games;

    high_resolution_clock::time_point start;
    uint64_t timeInMicro;
};

#endif // MCTSCPU_H
