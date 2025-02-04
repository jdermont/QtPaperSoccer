#ifndef NEGAMAXCPU_H
#define NEGAMAXCPU_H

#define MAX_GOAL 1000
#define MAX_ONE_EMPTY 850
#define MAX_CUTOFF 800
#define MIN_CUTOFF -800
#define MIN_GOAL_ONE_EMPTY -850
#define MIN_GOAL_NEXT_MOVE -900
#define MIN_GOAL -950
#define MIN_BLOCKED -1000
#define TERMINAL_THRESHOLD 500

#define LIMIT_MOVES 100

#include <algorithm>
#include <chrono>
#include <vector>
#include <deque>

#include "cpu.h"
#include "pitch.h"
#include "game.h"
#include "network.h"

using namespace std;
using namespace std::chrono;

class NegamaxMove : public IMove {
public:
    float score;

    NegamaxMove(string move, float score) : IMove(move), score(score) {

    }

    bool operator < (const NegamaxMove& move) const {
        return score < move.score;
    }
    bool operator > (const NegamaxMove& move) const {
        return score > move.score;
    }
};

class Evaluator {
public:
    Evaluator() {

    }

    virtual float evaluate(Game *game, player_t player) {
        int n = game->pitch.ball;
        int score = player == ONE ? game->pitch.getPosition(n).y : 10 - game->pitch.getPosition(n).y;
        return score;
    }
};

class NoEvaluator : public Evaluator {
public:
    NoEvaluator() : Evaluator() {

    }

    float evaluate(Game *game, player_t player) override {
        return 0;
    }
};

class BackerEvaluator : public Evaluator {
public:
    BackerEvaluator() : Evaluator() {

    }

    float evaluate(Game *game, player_t player) override {
        int n = game->pitch.ball;
        int score = player == ONE ? 10 - game->pitch.getPosition(n).y : game->pitch.getPosition(n).y;
        //        int score = game->pitch.getDistancesToGoal(player);
        return score;
    }
};

class SmartDistanceEvaluator : public Evaluator {
public:
    SmartDistanceEvaluator() : Evaluator() {

    }

    float evaluate(Game *game, player_t player) override {
        int score = game->pitch.getDistancesToGoal(player);
        return score;
    }
};

class SmartBackerDistanceEvaluator : public Evaluator {
public:
    SmartBackerDistanceEvaluator() : Evaluator() {

    }

    float evaluate(Game *game, player_t player) override {
        int score = -game->pitch.getDistancesToGoal(player);
        return score;
    }
};

class NetworkEvaluator : public Evaluator {
public:
    Network *agent;
    Random ran;

    NetworkEvaluator(Network *agent) : Evaluator(), agent(agent) {

    }

    float evaluate(Game *game, player_t player) override {
        auto tuples = game->getTuplesEdges3();
        float score = agent->getScore(tuples);
        score = 0.95f * score + 0.05f * ran.nextFloat(-1,1);
        return player == game->currentPlayer ? score : -score;
    }
};

class NegamaxCpu : public ICpu {
public:
    NegamaxCpu(Evaluator* evaluator = new Evaluator(), int limit = LIMIT_MOVES) : ICpu(), evaluator(evaluator), limit(limit) {

    }

    NegamaxMove getBestMove(uint64_t timeInMicro, int levels = 50) {
        start = high_resolution_clock::now();
        measureTime = true;
        this->timeInMicro = timeInMicro;
        alreadyBlocking = game->pitch.isCutOffFromOpponentGoal(player == ONE ? TWO : ONE);
        alreadyBlocked = game->pitch.isCutOffFromOpponentGoal(player);

        vector<NegamaxMove> moves = generateMoves();
        NegamaxMove bestMove = getBestMoveSoFar(moves);

        if (bestMove.score > TERMINAL_THRESHOLD) {
            return bestMove;
        }

        for (int i=1; i < levels; i++) {
            for (auto & move : moves) {
                if (move.score < -TERMINAL_THRESHOLD) {
                    continue;
                }
                auto paths = makeMove(move.move);
                game->changePlayer();
                game->rounds++;
                try {
                    move.score = -getScore(i,-1, i-1, -INF, INF);
                } catch (int e) {
                    game->rounds--;
                    game->changePlayer();
                    undoMove(paths);
                    return bestMove;
                }
                game->rounds--;
                game->changePlayer();
                undoMove(paths);
                if (move.score > bestMove.score) {
                    bestMove = move;
                }
            }
            bestMove = getBestMoveSoFar(moves);
            if (abs(bestMove.score) > TERMINAL_THRESHOLD) {
                return bestMove;
            }
        }

        return bestMove;
    }

private:
    Evaluator* evaluator;
    int limit;
    bool measureTime;
    high_resolution_clock::time_point start;
    uint64_t timeInMicro;
    bool alreadyBlocking;
    bool alreadyBlocked;

    NegamaxMove getBestMoveSoFar(vector<NegamaxMove> & moves) {
        sort(moves.rbegin(), moves.rend());

        float score = moves[0].score;
        int n = 0;
        for (auto & move : moves) {
            if (move.score < score) break;
            n++;
        }

        return moves[ran.nextInt(n)];
    }

    vector<NegamaxMove> generateMoves() {
        vector<NegamaxMove> moves; moves.reserve(50);
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
            check = game->pitch.shouldCheckForGameOver(this->player);
        }

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
                        if (goal == this->player) {
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
                                score = evaluator->evaluate(game, this->player);
                                game->changePlayer();
                                game->rounds--;
                                //score = this->player == ONE ? 10 - game->pitch.getPosition(n).y : game->pitch.getPosition(n).y;
                            }
                        }
                    }
                    game->pitch.ball = t;
                    game->pitch.removeEdge(t, n);
                    str += game->pitch.getDistanceChar(t, n);

                    moves.push_back(NegamaxMove(str, score));
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

    float getScore(int levels, int color, int level, float alpha, float beta) {
        if (measureTime) {
            high_resolution_clock::time_point stop = high_resolution_clock::now();
            if (duration_cast<microseconds>(stop - start).count() >= timeInMicro) {
                throw -1;
            }
        }

        player_t player = color == 1 ? this->player : (this->player == ONE ? TWO : ONE);

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

        bool alreadyBlocking = color == 1 ? this->alreadyBlocking : this->alreadyBlocked;
        bool alreadyBlocked = color == 1 ? this->alreadyBlocked : this->alreadyBlocking;

        int loop = 0;

        float output = -INF;

        int count = 0;
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
                                if (level > 0) {
                                    game->changePlayer();
                                    game->rounds++;
                                    try {
                                        score = -getScore(levels,-color, level-1, -beta, -alpha);
                                    } catch (int e) {
                                        game->rounds--;
                                        for (int i = 0; i < konceEdges.size() / 2; i++) {
                                            game->pitch.removeEdge(konceEdges[2 * i], konceEdges[2 * i + 1]);
                                        }
                                        game->changePlayer();
                                        game->pitch.removeEdge(t, n);
                                        for (auto &p : paths) game->pitch.removeEdge(p.a, p.b);
                                        game->pitch.ball = vertices[0];
                                        throw e;
                                    }
                                    game->changePlayer();
                                    game->rounds--;
                                } else {
                                    game->changePlayer();
                                    score = color * evaluator->evaluate(game, this->player);
                                    game->changePlayer();
                                    //score = this->player == ONE ? 10 - game->pitch.getPosition(n).y : game->pitch.getPosition(n).y;
                                }
                            }
                        }
                    }
                    // game->changePlayer();
                    game->pitch.ball = t;
                    game->pitch.removeEdge(t, n);
                    output = max(score, output);
                    alpha = max(output, alpha);
                    if (alpha >= beta) {
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

#endif // NEGAMAXCPU_H
