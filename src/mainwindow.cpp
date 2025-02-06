#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    centralWidget()->setAttribute(Qt::WA_TransparentForMouseEvents);
    setMouseTracking(true);
    QPushButton* button = new QPushButton(QString("Back"), this);
    button->setVisible(true);

    this->resize(480,640);

    QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
    int poolSize = settings.value("poolSize", 1<<24).toInt();
    settings.setValue("poolSize",poolSize);
    int moveLimit = settings.value("moveLimit", 750).toInt();
    settings.setValue("moveLimit",moveLimit);
    bool kurnikColors = settings.value("kurnikColors", false).toBool();
    settings.setValue("kurnikColors",kurnikColors);
    int hidden = settings.value("hidden", 96).toInt();
    settings.setValue("hidden",hidden);
    int hidden2 = settings.value("hidden2", 32).toInt();
    settings.setValue("hidden2",hidden2);
    QString netfile = settings.value("netfile","96_32_net").toString();
    settings.setValue("netfile",netfile);
    cpuParallel = new CpuMctsTTRParallel(poolSize);
    cpuParallel->moveLimit = moveLimit;
    Network* network = new NetworkDeep(1466,hidden,hidden2);
    network->load(netfile.toStdString());
    network->type = 2;
    cpuParallel->agent = network;

    qRegisterMetaType<string>("string");

    connect(button, SIGNAL (released()),this, SLOT (onBack()));
    connect(this, SIGNAL(sendMessage(string)), &secondWindow, SLOT(onMessage(string)));
    connect(this, SIGNAL(sendNotation(string)), &secondWindow, SLOT(onNotation(string)));
    connect(this, SIGNAL(sendNotation2(string)), &secondWindow, SLOT(onNotation2(string)));
    connect(this, SIGNAL(sendGameState(bool)), &secondWindow, SLOT(onGameStateChanged(bool)));
    connect(this, SIGNAL(sendWinner(int)), &secondWindow, SLOT(onWinner(int)));
    connect(&secondWindow, SIGNAL(startClicked(string)), this, SLOT(onStartClicked(string)));
    secondWindow.show();
}

void MainWindow::onBack() {
    if (!game->started || calculating || (game->isOver() && !game->almost)) return;
    if (game->notation.size() == 0) {
        return;
    }
    if (game->notation.back() == ',') {
        QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
        if (settings.value("computer",true).toBool() && game->currentPlayer == TWO) {
            game->undoMove();
            while (game->notation.size() > 0 && game->notation.back() != ',') {
                game->undoMove();
            }
            if (game->notation.size() > 0) {
                game->undoChangePlayer();
            }
            if (game->notation.empty()) calcMove(false);
        } else {
            game->undoChangePlayer();
        }
    } else {
        game->undoMove();
    }
    emit sendNotation(game->notation);
    repaint();
    return;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::mouseMoveEvent(QMouseEvent *event) {
    double x = (event->x() - marginWidth) / blocksize;
    double y = (event->y() - marginHeight) / blocksize;
    hoverX = round(x);
    hoverY = round(y);
    repaint();
}

void MainWindow::keyPressEvent(QKeyEvent *event) {
    // if (event->key() == Qt::Key_F) {
    //     flip = !flip;
    //     repaint();
    //     return;
    // }
    if (!game->started || calculating || (game->isOver() && !game->almost)) return;
    if (event->key() == Qt::Key_Backspace) {
        if (game->notation.size() == 0) {
            return;
        }
        if (game->notation.back() == ',') {
            QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
            if (settings.value("computer",true).toBool() && game->currentPlayer == TWO) {
                game->undoMove();
                while (game->notation.size() > 0 && game->notation.back() != ',') {
                    game->undoMove();
                }
                if (game->notation.size() > 0) {
                    game->undoChangePlayer();
                }
                if (game->notation.empty()) calcMove(false);
            } else {
                game->undoChangePlayer();
            }
        } else {
            game->undoMove();
        }
        emit sendNotation(game->notation);
        repaint();
        return;
    }
    if (game->isOver() || !game->started || game->almost || calculating) return;
    if (event->key() == Qt::Key_Shift) {
        calcMove(true);
        return;
    }
    return;

    if (event->key() == Qt::Key_C) {
        game->changePlayer();
        repaint();
        return;
    }
    if (event->key() == Qt::Key_Space) {
        delete game;
        game = new Game();
        repaint();
        return;
    }
    if (game->isOver()) return;
    if (event->key() == Qt::Key_Q) {
        if (game->currentPlayer == ONE || false) {
            Network network(435,320);
            network.loadCheckpoint("RL1");
            network.type = 1;
            CpuMctsTTR3 cpu(1<<21); cpu.setGame(game); cpu.setPlayer(game->currentPlayer); cpu.agent = &network;
            auto move = cpu.getBestMove(100 * 1000, true);
            cout << "games " << cpu.games << ", maxLevel: " << cpu.maxLevel << endl;
            cout << move->score / move->games << " ::: " << move->heuristic << endl;
            game->makeMove(move->move,game->currentPlayer);
        } else {
            MctsCpu cpu; cpu.setGame(game); cpu.setPlayer(game->currentPlayer);
            auto move = cpu.getBestMove(500 * 1000);
            game->makeMove(move->move,game->currentPlayer);
            delete move;
        }
    }
    repaint();
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event) {
    if (game->isOver() && !game->almost) return;
    if (!game->started) return;
    if (calculating) return;

    QPoint p(hoverX, hoverY);
    Point bP = game->pitch.getPosition(game->pitch.ball);
    if (game->almost) {
        if (p.x() == bP.x && p.y() == bP.y) {
            game->confirm();
            emit sendNotation(game->notation);
            if (game->isOver()) {
                sendGameState(!game->isOver());
                emit sendWinner(game->getWinner() == humanPlayer ? 0 : 1);
            } else {
                if (game->currentPlayer == ONE) {
                    QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
                    if (settings.value("computer",true).toBool()) {
                        calcMove(false);
                    }
                }
            }
        }
        return;
    }
    if (abs(p.x()-bP.x) <= 1 && abs(p.y()-bP.y) <= 1 && game->pitch.getNeighbour(p.x()-bP.x,p.y()-bP.y) != -1) {
        if (p.x() - bP.x == -1) {
            if (p.y() - bP.y == -1) game->makeMoveWithPlayer('7');
            else if (p.y() - bP.y == 0) game->makeMoveWithPlayer('6');
            else game->makeMoveWithPlayer('5');
        } else if (p.x() - bP.x == 0) {
            if (p.y() - bP.y == -1) game->makeMoveWithPlayer('0');
            else game->makeMoveWithPlayer('4');
        } else {
            if (p.y() - bP.y == -1) game->makeMoveWithPlayer('1');
            else if (p.y() - bP.y == 0) game->makeMoveWithPlayer('2');
            else game->makeMoveWithPlayer('3');
        }
        emit sendNotation(game->notation);
    }
    repaint();
}

void MainWindow::paintEvent(QPaintEvent *event) {
    QWidget::paintEvent(event);

    // calculating size
    QRect rect = event->rect();
    int width = rect.width();
    int height = rect.height();
    blocksize = (double)height / (game->pitch.height+3);
    marginWidth = (width - blocksize * game->pitch.width) / 2.0f;
    if (marginWidth <= 0) {
        blocksize = (double)width / (game->pitch.width);
        marginWidth = (width - blocksize * game->pitch.width) / 2.0f;
    }
    marginHeight = (1.5f * blocksize);

    QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
    bool kurnikColors = settings.value("kurnikColors", false).toBool();
    //settings.setValue("kurnikColors",kurnikColors);

    // paint stuff
    QPen white(QColor("#FFFFFF"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
    QPen red(QColor("#FF0000"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
    QPen gray(QColor("#D0D0D0"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
    QPen black(QColor("#000000"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
    QPen blue(QColor("#0000FF"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
    QPen fieldLine(QColor("#CCCCEE"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);

    if (kurnikColors) {
        white = QPen(QColor("#308048"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
        red = QPen(QColor("#F0F0F0"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
        blue = QPen(QColor("#E0E0E0"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
        black = QPen(QColor("#FFFFFF"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
        fieldLine = QPen(QColor("#34844B"), LINE_WIDTH, Qt::SolidLine, Qt::RoundCap);
    }

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    QFont font;
    int i;

    font.setPixelSize(blocksize/3);
    painter.setFont(font);

    // background
    painter.fillRect(rect,QColor(kurnikColors ? "#308048" : "#FFFFFF"));

    // background lines
    painter.setPen(fieldLine);
    i = 0;
    while (marginWidth + i * blocksize >= 0) {
        painter.drawLine(marginWidth+blocksize*i,0,marginWidth+blocksize*i,height);
        i--;
    }
    i = 1;
    while (marginWidth + i * blocksize <= width) {
        painter.drawLine(marginWidth+blocksize*i,0,marginWidth+blocksize*i,height);
        i++;
    }
    i = 0;
    while (marginHeight + i * blocksize >= 0) {
        painter.drawLine(0,marginHeight+blocksize*i,width,marginHeight+blocksize*i);
        i--;
    }
    i = 1;
    while (marginHeight + i * blocksize <= height) {
        painter.drawLine(0,marginHeight+blocksize*i,width,marginHeight+blocksize*i);
        i++;
    }

    red.setWidth(LINE_WIDTH);
    blue.setWidth(LINE_WIDTH);
    black.setWidth(LINE_WIDTH);

    Pitch pitch = game->pitch;

    // for (auto & edge : game->pitch.getAllEdges()) {
    //     painter.setPen(gray);
    //     Point a = pitch.getPosition(edge.a);
    //     Point b =  pitch.getPosition(edge.b);
    //     painter.drawLine(marginWidth+blocksize*a.x,marginHeight+blocksize*a.y,marginWidth+blocksize*b.x,marginHeight+blocksize*b.y);
    // }

    if (flip) {
        for (auto & edge : pitch.getEdgesMirrored()) {
            if (edge.player == NONE) {
                painter.setPen(black);
            } else if (edge.player == TWO) {
                painter.setPen(red);
            } else {
                painter.setPen(blue);
            }
            painter.drawLine(marginWidth+blocksize*edge.a.x,marginHeight+blocksize*edge.a.y,marginWidth+blocksize*edge.b.x,marginHeight+blocksize*edge.b.y);
        }
    } else {
        for (auto & edge : pitch.getEdges()) {
            if (edge.player == NONE) {
                painter.setPen(black);
            } else if (edge.player == TWO) {
                painter.setPen(red);
            } else {
                painter.setPen(blue);
            }
            painter.drawLine(marginWidth+blocksize*edge.a.x,marginHeight+blocksize*edge.a.y,marginWidth+blocksize*edge.b.x,marginHeight+blocksize*edge.b.y);
        }
    }



       // red.setWidth(BOLD_LINE_WIDTH);
       // blue.setWidth(BOLD_LINE_WIDTH);

       // if (game->currentPlayer == ONE) {
       //     painter.setPen(red);
       // } else {
       //     painter.setPen(blue);
       // }

       // for (auto& path : game->paths) {
       //     Point a = pitch.getPosition(path.a);
       //     Point b = pitch.getPosition(path.b);
       //     Edge edge = Edge(a,b);
       //     painter.drawLine(marginWidth+blocksize*edge.a.x,marginHeight+blocksize*edge.a.y,marginWidth+blocksize*edge.b.x,marginHeight+blocksize*edge.b.y);
       // }

    if (game->almost) {
        red.setWidth(BIG_BALL_SIZE);
        blue.setWidth(BIG_BALL_SIZE);
        black.setWidth(BIG_BALL_SIZE);
    } else {
        red.setWidth(BALL_SIZE);
        blue.setWidth(BALL_SIZE);
        black.setWidth(BALL_SIZE);
    }

    // painter.setPen(gray);
    // for (int i=0;i<pitch.size;i++) {
    //     Point p = pitch.getPosition(i);
    //     painter.drawText(marginWidth+2+blocksize*p.x,-2+marginHeight+blocksize*p.y,QString::number(i));
    // }

    painter.setPen(game->currentPlayer==TWO?red:blue);
    Point ball = !flip ? pitch.getPosition(pitch.ball) : pitch.getPosition(pitch.getMirroredIndex(pitch.ball));
    painter.drawPoint(marginWidth+blocksize*ball.x,marginHeight+blocksize*ball.y);

    //    for (int i=0;i<pitch.cutOffGoalArray.size();i++) {
    //        if (pitch.cutOffGoalArray[i] == ONE) {
    //            painter.setPen(red);
    //            Point ball = pitch.getPosition(i);
    //            painter.drawPoint(marginWidth+blocksize*ball.x,marginHeight+blocksize*ball.y);
    //        } else if (pitch.cutOffGoalArray[i] == TWO) {
    //            painter.setPen(blue);
    //            Point ball = pitch.getPosition(i);
    //            painter.drawPoint(marginWidth+blocksize*ball.x,marginHeight+blocksize*ball.y);
    //        }
    //    }

    red.setWidth(BALL_SIZE);
    blue.setWidth(BALL_SIZE);
    black.setWidth(BALL_SIZE);

    //    if (hoverX >= 0 && hoverY >= -1) {
    painter.setPen(black);
    painter.drawPoint(marginWidth+blocksize*hoverX,marginHeight+blocksize*hoverY);
    //    }

    black.setWidth(LINE_WIDTH);
    painter.setPen(black);
}

void MainWindow::onMoveCalculated(char c) {
    game->makeMoveWithPlayer(c);
    emit sendNotation(game->notation);
    if (game->almost) {
        game->confirm();
        calculating = false;
    }
    if (game->isOver()) {
        sendGameState(!game->isOver());
        emit sendWinner(game->getWinner() == humanPlayer ? 0 : 1);
    }
    repaint();
}

void MainWindow::onMoveCalculated2(char c) {
    game->makeMoveWithPlayer(c);
    emit sendNotation(game->notation);
    if (game->almost) {
        calculating = false;
    }
    repaint();
}

void MainWindow::onStartClicked(string notation) {
    if (calculating) return;
    delete game;
    game = new Game();
    game->started = true;
    bool cpuStart = false;
    if (notation.size() > 0) {
        game->currentPlayer = TWO;
        for (auto & c : notation) {
            if (c < '0' || c > '7') continue;
            game->makeMoveWithPlayer(c,true);
        }
        humanPlayer = game->currentPlayer;
    } else {
        humanPlayer = humanPlayer == ONE ? TWO : ONE;
        if (humanPlayer == TWO) game->currentPlayer = TWO;
        cpuStart = true;
    }
    emit sendMessage("Game started");
    emit sendNotation(game->notation);
    repaint();
    emit sendGameState(true);
    if (cpuStart) {
        if (game->currentPlayer == ONE) {
            QSettings settings("qtpapersoccer.ini", QSettings::IniFormat);
            if (settings.value("computer",true).toBool()) {
                calcMove(false);
            }
        }
    }
}

void MainWindow::calcMove(bool forHuman) {
    calculating = true;
    if (forHuman) {
        emit sendMessage("Calculating move for human...");
    } else {
        emit sendMessage("Calculating move...");
    }
    cpuParallel->setGame(game);
    cpuParallel->setPlayer(game->currentPlayer);
    // Create an instance of your woker
    WorkerThread *workerThread = new WorkerThread(game,cpuParallel);
    // Connect our signal and slot
    if (forHuman) {
        connect(workerThread, SIGNAL(moveCalculated(char)),
                SLOT(onMoveCalculated2(char)));
    } else {
        connect(workerThread, SIGNAL(moveCalculated(char)),
                SLOT(onMoveCalculated(char)));
    }
    connect(workerThread, SIGNAL(moveLogs(string)), &secondWindow, SLOT(onMoveLogs(string)));
    connect(workerThread, SIGNAL(finished()), workerThread,
            SLOT(deleteLater()));
    workerThread->start();
}
