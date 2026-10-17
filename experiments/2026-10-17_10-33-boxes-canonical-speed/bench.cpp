#include "boxes/boxes_solver.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Replays masks from completed play_game arena histories; never generates games.
// One thread, fresh table per root, then warm reuse over its legal children.
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1]);
    std::vector<uint64_t> masks;
    uint64_t mask;
    while (input >> mask) masks.push_back(mask);
    const std::string mode = argv[2];
    std::cout << std::setprecision(9);
    if (mode == "construct") {
        for (std::size_t entries : {std::size_t{2}, std::size_t{1} << 20}) {
            uint64_t checksum = 0;
            const auto start = std::clock();
            for (int repeat = 0; repeat < 1000; ++repeat) {
                alpha_go::BoxesSolver solver(5, 5, entries);
                checksum += solver.table_bytes();
            }
            std::cout << "{\"cpu_seconds\":"
                      << double(std::clock() - start) / CLOCKS_PER_SEC
                      << ",\"entries\":" << entries
                      << ",\"checksum\":" << checksum << "}\n";
        }
        return 0;
    }
    if (mode == "canonical") {
        alpha_go::BoxesSolver solver(5, 5, 2);
        uint64_t checksum = 0;
        const auto start = std::clock();
        for (int repeat = 0; repeat < 100000; ++repeat) {
            for (uint64_t m : masks) checksum += solver.canonical(m);
        }
        std::cout << "{\"cpu_seconds\":"
                  << double(std::clock() - start) / CLOCKS_PER_SEC
                  << ",\"checksum\":" << checksum << "}\n";
        return 0;
    }
    for (std::size_t root = 0; root < masks.size(); ++root) {
        alpha_go::BoxesSolver solver(5, 5, std::size_t{1} << 20);
        for (int edge = -1; edge < 60; ++edge) {
            if (edge >= 0 && (masks[root] & (uint64_t{1} << edge))) continue;
            const uint64_t child = masks[root] | (edge < 0 ? 0 : uint64_t{1} << edge);
            const auto start = std::clock();
            const auto value = solver.value_within(child, 50000);
            const double seconds = double(std::clock() - start) / CLOCKS_PER_SEC;
            std::cout << "{\"root\":" << root << ",\"edge\":" << edge
                      << ",\"value\":" << (value ? std::to_string(*value) : "null")
                      << ",\"nodes\":" << solver.nodes()
                      << ",\"cpu_seconds\":" << seconds << "}\n";
        }
    }
}
