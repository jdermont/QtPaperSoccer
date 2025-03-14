#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QPaintEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QRect>
#include <QPen>
#include <QBrush>
#include <QFont>
#include <QFontMetrics>
#include <math.h>
#include <QThread>
#include <QPushButton>

#define LINE_WIDTH 2
#define BOLD_LINE_WIDTH 3
#define BALL_SIZE 8
#define BIG_BALL_SIZE 14

#include "game.h"
#include "cpu.h"
#include "cpumctstt.h"
#include "mctscpu.h"
#include "secondwindow.h"
#include "workerthread.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    virtual void paintEvent(QPaintEvent *);
    virtual void mouseReleaseEvent(QMouseEvent *);
    virtual void mouseMoveEvent(QMouseEvent *);
    virtual void keyPressEvent(QKeyEvent *);

    bool flip = false;

private:
    Ui::MainWindow *ui;
    SecondWindow secondWindow;

    player_t humanPlayer = ONE;
    Game *game = new Game();
    CpuMctsTTRParallel* cpuParallel;

    double blocksize;
    double marginWidth;
    double marginHeight;

    int hoverX = -1;
    int hoverY = -1;

    bool calculating = false;
    void calcMove(bool forHuman);

public slots:
    void onMoveCalculated(char c);
    void onMoveCalculated2(char c);
    void onStartClicked(string notation);
    void onBack();


signals:
    void sendMessage(string message);
    void sendNotation(string notation);
    void sendNotation2(string notation);
    void sendGameState(bool inGame);
    void sendWinner(int winner);
};
#endif // MAINWINDOW_H
