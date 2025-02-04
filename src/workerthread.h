#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H

#include <QThread>
#include <memory>
#include <QSettings>
#include "game.h"
#include "cpumctstt.h"

using namespace std;

class WorkerThread : public QThread
{
    Q_OBJECT
public:
    WorkerThread(Game *game, CpuMctsTTRParallel* cpu) : cpu(cpu) {
        this->game.reset(new Game(*game));
        cpu->setGame(this->game.get());
    }
    virtual void run();

private:
    shared_ptr<Game> game;
    CpuMctsTTRParallel* cpu;

signals:
    void moveCalculated(char c);
    void moveLogs(const string & logs);
};

#endif // WORKERTHREAD_H
