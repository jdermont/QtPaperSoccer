#ifndef RL_H
#define RL_H

#include "utils.h"
#include "game.h"
#include "cpumctstt.h"
#include "mctscpu.h"
#include "negamaxcpu.h"
#include "network.h"
#include "random.h"

#include <thread>
#include <mutex>
#include <random>

using namespace std;
using namespace std::chrono;

Network *testNetwork;

#endif // RL_H
