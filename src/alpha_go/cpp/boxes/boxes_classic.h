#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "boxes/boxes_solver.h"

namespace alpha_go {

// Optional classical engine. All scores are remaining margins for the mover, in
// hundredths of a box; adding board.margin() gives a final-margin evaluation.
// A steady-clock deadline covers the entire iterative-deepening call. Exact-solver
// probes retain their table across 1024-node slices, checking the clock between
// slices, up to 16384 total nodes. One instance and one thread per player.
class BoxesClassic {
public:
    BoxesClassic(int rows, int cols = 0, double seconds = 1.0, int solver_max_undrawn = 26,
                 std::size_t table_entries = std::size_t{1} << 18,
                 std::size_t solver_entries = std::size_t{1} << 20);

    int best_edge(uint64_t mask, uint64_t seed = 0);
    double evaluate(uint64_t mask);
    double final_margin(const BoxesBoard& board) { return board.margin() + evaluate(board.edges()); }
    // Deterministic fixed-depth search for correctness checks, without a clock deadline.
    double value(uint64_t mask, int depth);
    int completed_depth() const { return completed_depth_; }
    double last_value() const { return last_value_ / 100.0; }
    uint64_t nodes() const { return nodes_; }
    uint64_t solver_nodes() const { return solver_nodes_; }
    static uint64_t probe_node_budget() { return kProbeBudget; }
    static uint64_t probe_slice_nodes() { return kProbeSlice; }
    std::size_t table_bytes() const { return table_.size() * sizeof(Entry) + solver_.table_bytes(); }

private:
    enum Flag : int8_t { kEmpty = 0, kExact = 1, kLower = 2, kUpper = 3 };
    static constexpr int kScale = 100;
    static constexpr int kInf = 10000;
    static constexpr int kSolved = 65;
    static constexpr uint64_t kProbeBudget = 16384;
    static constexpr uint64_t kProbeSlice = 1024;
    struct Entry {
        uint64_t key;
        int16_t value;
        int8_t depth;
        int8_t flag;
        int8_t move;
    };
    int estimate(uint64_t mask);
    int quiet_estimate(uint64_t mask, const std::vector<int>& deg,
                       const std::vector<chains::Component>& comps);
    int search(uint64_t mask, int depth, int alpha, int beta, int& move, bool root = false);
    bool expired();
    std::vector<int> moves(uint64_t mask, const std::vector<int>& deg,
                           const std::vector<chains::Component>& comps, int first) const;
    std::size_t index(uint64_t mask) const {
        return static_cast<std::size_t>((mask * 0x9E3779B97F4A7C15ULL) >> (64 - log2_size_));
    }

    BoxesSolver solver_;
    uint64_t full_;
    double seconds_;
    int solver_max_undrawn_;
    int log2_size_ = 1;
    std::vector<Entry> table_;
    std::vector<uint32_t> history_;
    std::vector<int> rank_;
    std::chrono::steady_clock::time_point deadline_;
    bool timed_ = false;
    bool aborted_ = false;
    uint64_t nodes_ = 0;
    uint64_t solver_nodes_ = 0;
    int completed_depth_ = 0;
    int last_value_ = 0;
};

}  // namespace alpha_go
