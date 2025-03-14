#ifndef SECONDWINDOW_H
#define SECONDWINDOW_H

#include <QMainWindow>
#include <string>
#include <iostream>
#include <QValidator>
#include <QThread>
#include <QSettings>

using namespace std;

namespace Ui {
class SecondWindow;
}

class SecondWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit SecondWindow(QWidget *parent = nullptr);
    ~SecondWindow();

private:
    Ui::SecondWindow *ui;
    int stats[2] = {};

public slots:
    void onMessage(string message);
    void onStartBtnClicked();
    void onMoveLogs(const string & logs);
    void onNotation(string notation);
    void onNotation2(string notation);
    void onGameStateChanged(bool inGame);
    void onTimeTextChanged(const QString & str);
    void onThreadsTextChange(const QString & str);
    void onWinner(int winner);
    void onStateChanged(int state);
    void onStateChanged2(int state);
    void onStateChanged3(int state);

signals:
    void startClicked(string notation);
};

#endif // SECONDWINDOW_H
