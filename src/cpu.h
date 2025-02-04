#ifndef CPU_H
#define CPU_H

#define INF 1000
#define TT_EXACT 0
#define TT_LOWER_BOUND 1
#define TT_UPPER_BOUND 2

#include "game.h"
#include "random.h"

#include <string>
#include <stdint.h>

using namespace std;

class IMove {
public:
    string move;
    IMove(string move) : move(move) {

    }

    virtual ~IMove() { }
};

class ICpu {
public:
    ICpu() {

    }

    void setGame(Game *game) {
        this->game = game;
    }

    void deleteGame() {
        delete game;
    }

    void setPlayer(player_t player) {
        this->player = player;
    }

    player_t getPlayer() {
        return player;
    }

    virtual ~ICpu() { }
protected:
    Game *game;
    player_t player;
    Random ran;

    // fisher yates
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

    int nextNode(char &c) {
        switch (c) {
        case '5':
            return game->pitch.getNeighbour(-1, 1);
        case '4':
            return game->pitch.getNeighbour(0, 1);
        case '3':
            return game->pitch.getNeighbour(1, 1);
        case '6':
            return game->pitch.getNeighbour(-1, 0);
        case '2':
            return game->pitch.getNeighbour(1, 0);
        case '7':
            return game->pitch.getNeighbour(-1, -1);
        case '0':
            return game->pitch.getNeighbour(0, -1);
        case '1':
            return game->pitch.getNeighbour(1, -1);
        }
        return -1;
    }
};

class TTEntry {
public:
    int value;
    int flag;
};


#endif // CPU_H
