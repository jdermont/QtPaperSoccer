#include "secondwindow.h"
#include "ui_secondwindow.h"

SecondWindow::SecondWindow(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::SecondWindow)
{
    ui->setupUi(this);
    QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
    int time = settings.value("time", 1).toInt();
    int threads = settings.value("threads", 4).toInt();
    bool comp = settings.value("computer",true).toBool();
    ui->timeEdit->setValidator(new QIntValidator(1, 600, this));
    ui->timeEdit->setText(QString::number(time));
    ui->threadEdit->setValidator(new QIntValidator(1, 8, this));
    ui->threadEdit->setText(QString::number(threads));
    ui->checkBox->setChecked(comp);
    connect(ui->startBtn, SIGNAL (released()),this, SLOT (onStartBtnClicked()));
    connect(ui->timeEdit, SIGNAL (textChanged(const QString &)),this, SLOT (onTimeTextChanged(const QString &)));
    connect(ui->threadEdit, SIGNAL (textChanged(const QString &)),this, SLOT (onThreadsTextChange(const QString &)));
    connect(ui->checkBox, SIGNAL (stateChanged(int)),this, SLOT (onStateChanged(int)));
}

SecondWindow::~SecondWindow()
{
    delete ui;
}

void SecondWindow::onMessage(string message) {
    ui->textEdit->append(QString::fromStdString(message));
}

void SecondWindow::onStartBtnClicked() {
    emit startClicked(ui->notationEdit->text().toStdString());
}

void SecondWindow::onMoveLogs(const string & logs) {
    ui->textEdit->append(QString::fromStdString(logs));
}

void SecondWindow::onNotation(string notation) {
    ui->lineEdit->setText(QString::fromStdString(notation));
}

void SecondWindow::onNotation2(string notation) {
    ui->lineEdit_2->setText(QString::fromStdString(notation));
}

void SecondWindow::onGameStateChanged(bool inGame) {
    ui->startBtn->setEnabled(!inGame);
}

void SecondWindow::onTimeTextChanged(const QString & str) {
    QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
    settings.setValue("time", str);
}

void SecondWindow::onThreadsTextChange(const QString & str) {
    QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
    settings.setValue("threads", str);
}

void SecondWindow::onWinner(int winner) {
    stats[(winner+1)&1]++;
    ui->statsLabel->setText(QString::number(stats[0])+" : "+QString::number(stats[1]));
}

void SecondWindow::onStateChanged(int state) {
    bool checked = state != 0;
    QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
    settings.setValue("computer", checked);
}
