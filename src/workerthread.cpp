#include "workerthread.h"

void WorkerThread::run() {
    QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
    int threads = settings.value("threads", 4).toInt();
    if (threads < 1) threads = 1;
    if (threads > 8) threads = 8;
    int time = settings.value("time", 1).toInt();
    if (time == 0) time = 1;
    string move;
    if (game->pitch.isNextMoveGameover(cpu->getPlayer()==ONE?TWO:ONE)) {
        move = game->pitch.shortWinningMoveForPlayer(cpu->getPlayer());
        string logs = "short winning move "+move;
        emit moveLogs(logs);
    } else {
        move = cpu->getBestMove(time * 1000000L, threads)->move;
        emit moveLogs(cpu->ss.str());
    }


    for (auto & c : move) {
        emit moveCalculated(c);
        QThread::msleep(250);
    }
}
