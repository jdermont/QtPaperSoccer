#include "mainwindow.h"
#include "rl.h"

#include <QApplication>

string clop(int player, float alpha, float FPU, float C, float Croot) {
    CpuMctsTTRParallel cpu1(1<<25); cpu1.setPlayer(player == 1 ? ONE : TWO);
    cpu1.id = 0;
    cpu1.agent = testNetwork;
    cpu1.moveLimit = 750;
    CpuMctsTTRParallel cpu2(1<<25); cpu2.setPlayer(player == 0 ? ONE : TWO);
    cpu2.id = 0;
    cpu2.agent = testNetwork;
    cpu2.moveLimit = 750;

    cpu1.alpha = 0.4f;
    cpu1.FPU = 0.3f;
    cpu1.C = 0.5f;

    cpu2.alpha = alpha;
    cpu2.FPU = FPU;
    cpu2.C = C;
    cpu2.Croot = Croot;

    Game game;
    cpu1.setGame(&game);
    cpu2.setGame(&game);
    Random ran;//(Random().nextLong() ^ random_device()());
    while (!game.isOver()) {
        if (game.rounds == 0) {
            string t = std::to_string(ran.nextInt(8));
            game.makeMove(t);
            continue;
        }

        if (cpu1.getPlayer() == game.currentPlayer) {
            auto move = cpu1.getBestMove(1000000,3)->move;
            game.makeMove(move);
        } else {
            auto move = cpu2.getBestMove(1000000,3)->move;
            game.makeMove(move);
        }
    }

    int winner = game.getWinner();
    if (winner == cpu2.getPlayer()) {
        return "W";
    } else if (winner != NONE) {
        return "L";
    }
    return "D";
}

// #include <signal.h>

// MainWindow* mainWindow;


int main(int argc, char *argv[]) {
    srand(time(NULL) ^ uint64_t(&main));
    Pitch(8,10).GENERATE_ALL_EDGES();

    // signal(SIGUSR1, signalHandler);
    // cout << "pid " << getpid() << endl;

    // doRL();

    // int player;
    // float alpha,FPU,C,Croot;

    // for (int i=0; i < argc; i++) {
    //     // cout << i << " " << argv[i] << endl;
    //     player = atoi(argv[2]) % 2;
    //     alpha = atof(argv[4]);
    //     FPU = atof(argv[6]);
    //     C = atof(argv[8]);
    //     Croot = atof(argv[10]);
    // }

    // testNetwork = new NetworkScrelu(1466,96);
    // testNetwork->load("RL87");
    // testNetwork->type = 2;

    // cout << clop(player,alpha,FPU,C,Croot) << endl;
    // return 0;

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    // mainWindow = &w;
    return a.exec();
}
