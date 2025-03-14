#ifndef GAME_H
#define GAME_H

#include <string>
#include "pitch.h"

class Game {
public:
    player_t currentPlayer;

    Pitch pitch;
    string notation;
    int rounds;
    vector<Path> paths;
    bool started = false;
    bool almost = false;

    explicit Game() : currentPlayer(ONE), pitch(Pitch(8,10)) {
        rounds = 0;
    }

    void setGame(Game *game) {
        pitch.setPitch(game->pitch);
        currentPlayer = game->currentPlayer;
        rounds = game->rounds;
    }

    vector<string> getAvailableMoves() {
        vector<string> moves; moves.reserve(8);
        vector<int> ns(8);
        pitch.fillFreeNeighbours(ns, pitch.ball);
        for (auto & n : ns) {
            string move = ""; move += pitch.getDistanceChar(pitch.ball, n);
            moves.push_back(move);
        }
        return moves;
    }

    vector<int> getTuplesEdges() {
        return pitch.getTuplesEdges(currentPlayer);
    }

    vector<int> getTuplesEdgesBase() {
        return pitch.getTuplesEdgesBase(currentPlayer == ONE ? TWO : ONE);
    }

    vector<int> getTuplesEdges2() {
        return pitch.getTuplesEdges2(currentPlayer);
    }

    vector<int> getTuplesEdgesBase2() {
        return pitch.getTuplesEdgesBase2(currentPlayer == ONE ? TWO : ONE);
    }

    vector<int> getTuplesEdges3() {
        return pitch.getTuplesEdges3(currentPlayer);
    }

    vector<int> getTuplesEdgesBase3() {
        return pitch.getTuplesEdgesBase3(currentPlayer == ONE ? TWO : ONE);
    }

    vector<int> getTuplesEdges4() {
        return pitch.getTuplesEdges4(currentPlayer);
    }

    vector<int> getTuplesEdgesBase4() {
        return pitch.getTuplesEdgesBase4(currentPlayer == ONE ? TWO : ONE);
    }

    vector<int> getLegalActions() {
        vector<int> actions; actions.reserve(8);
        for (int i=0; i < 8; i++) {
            int n = nextNode(i+'0');
            if (n != -1 && !pitch.existsEdge(pitch.ball,n)) {
                actions.push_back(i);

            }
        }
        return actions;
    }

    char map(char c) {
        switch (c) {
        case '0': return '4';
        case '1': return '5';
        case '2': return '6';
        case '3': return '7';
        case '4': return '0';
        case '5': return '1';
        case '6': return '2';
        case '7': return '3';
        }
        return -1;
    }

    void makeMove(const string & move) {
        for (auto &c : move) {
            int n = nextNode(c);
            pitch.addEdge(pitch.ball,n);
            paths.push_back(Path(pitch.ball,n));
            pitch.ball = n;
        }
        notation += move;
        if (!isOver()) {
            if (!pitch.passNextDone(pitch.ball)) {
                changePlayer();
                notation += ",";
            }
        } else {
            notation += getWinner() == ONE ? " A":" B";
        }
        rounds++;
    }

    void makeMove(const string & move, int player) {
        for (auto &c : move) {
            int n = nextNode(c);
            pitch.addEdge(pitch.ball,n,player);
            paths.push_back(Path(pitch.ball,n));
            pitch.ball = n;
        }
        notation += move;
        if (!isOver()) {
            if (!pitch.passNextDone(pitch.ball)) {
                changePlayer();
                notation += ",";
            }
        } else {
            notation += getWinner() == ONE ? " A":" B";
        }
        rounds++;
    }

    void makeMove(char c) {
        int n = nextNode(c);
        pitch.addEdge(pitch.ball,n);
        paths.push_back(Path(pitch.ball,n));
        pitch.ball = n;
        notation += c;
        if (!isOver()) {
            if (!pitch.passNextDone(pitch.ball)) {
                changePlayer();
                notation += ",";
            }
        } else {
            notation += getWinner() == ONE ? " A":" B";
        }
        rounds++;
    }

    void confirm() {
        if (isOver()) {

        } else {
            changePlayer();
            notation += ",";
        }
        almost = false;
        rounds++;
    }

    void makeMoveWithPlayer(char c, bool fromNotation = false) {
        int n = nextNode(c);
        pitch.addEdge(pitch.ball,n,currentPlayer);
        paths.push_back(Path(pitch.ball,n));
        pitch.ball = n;
        notation += c;
        if (!isOver()) {
            if (!pitch.passNextDone(pitch.ball)) {
                if (fromNotation) {
                    changePlayer();
                    notation += ",";
                    rounds++;
                } else {
                    almost = true;
                }
            }
        } else {
            if (fromNotation) {
                changePlayer();
                notation += ",";
                rounds++;
            } else {
                almost = true;
            }
        }
    }

    void undoMove() {
        if (notation.size() > 0 && notation.back() == ',') {
            changePlayer();
            notation.pop_back();
            rounds--;
        }
        if (paths.size() > 0) {
            Path path = paths[paths.size()-1];
            paths.pop_back();
            pitch.removeEdge(path.a,path.b);
            pitch.ball = path.a;
            notation.pop_back();
        }
        almost = false;
    }

    void undoChangePlayer(bool needToConfirmMoves) {
        if (notation.size() > 0 && notation.back() == ',') {
            changePlayer();
            notation.pop_back();
            rounds--;
        }
        if (needToConfirmMoves) {
            almost = true;
        } else {
            while (notation.size() > 0 && notation.back() != ',') {
                if (paths.size() > 0) {
                    Path path = paths[paths.size()-1];
                    paths.pop_back();
                    pitch.removeEdge(path.a,path.b);
                    pitch.ball = path.a;
                    notation.pop_back();
                }
            }
            if (notation.size() > 0 && notation.back() == ',') {
                // notation.pop_back();
                // rounds--;
            }
        }
    }

    int nextNode(int index, char c) {
        switch(c) {
        case '5': return pitch.getNeighbour(index,-1,1);
        case '4': return pitch.getNeighbour(index,0,1);
        case '3': return pitch.getNeighbour(index,1,1);
        case '6': return pitch.getNeighbour(index,-1,0);
        case '2': return pitch.getNeighbour(index,1,0);
        case '7': return pitch.getNeighbour(index,-1,-1);
        case '0': return pitch.getNeighbour(index,0,-1);
        case '1': return pitch.getNeighbour(index,1,-1);
        }
        return -1;
    }

    int nextNode(char c) {
        switch(c) {
        case '5': return pitch.getNeighbour(-1,1);
        case '4': return pitch.getNeighbour(0,1);
        case '3': return pitch.getNeighbour(1,1);
        case '6': return pitch.getNeighbour(-1,0);
        case '2': return pitch.getNeighbour(1,0);
        case '7': return pitch.getNeighbour(-1,-1);
        case '0': return pitch.getNeighbour(0,-1);
        case '1': return pitch.getNeighbour(1,-1);
        }
        return -1;
    }

    void changePlayer() {
        currentPlayer = currentPlayer==ONE?TWO:ONE;
    }

    bool isOver() {
        return pitch.isBlocked(pitch.ball) || pitch.goal(pitch.ball)!=NONE;
    }

    player_t getWinner() {
        if (pitch.goal(pitch.ball)!=NONE) {
            if (pitch.goal(pitch.ball) == ONE) return TWO;
            return ONE;
        }
        if (currentPlayer == ONE) return TWO;
        return ONE;
    }

private:
};

#endif // GAME_H
