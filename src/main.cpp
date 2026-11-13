#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[]) {
    srand(time(NULL) ^ uint64_t(&main));
    Pitch(8,10).GENERATE_ALL_EDGES();

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    // mainWindow = &w;
    return a.exec();
}
