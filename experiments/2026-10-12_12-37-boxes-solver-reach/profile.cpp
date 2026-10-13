#include <fstream>
#include <iostream>
#include "boxes/boxes_solver.h"
int main(int argc, char** argv) {
    std::ifstream f(argv[1]);
    int n, count = 0;
    uint64_t mask;
    while (f >> n >> mask && count++ < 100) {
        alpha_go::BoxesSolver s(5, 5);
        std::cout << count << ' ' << s.value(mask) << ' ' << s.nodes() << '\n';
    }
}
