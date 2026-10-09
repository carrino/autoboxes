#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "boxes/boxes_game.h"

namespace alpha_go {

// Depth-limited alpha-beta baseline, the C++ port of alpha_go.boxes.agents.AlphaBeta
// (which stays as the reference). Negamax on the remaining box margin: capturing moves
// keep the turn and do not consume depth, a depth-0 position is scored by the greedy
// haul available to the side to move, moves are ordered captures / safe / loony after a
// seeded shuffle (so ties break randomly but the value is deterministic), and the table
// stores exact-depth entries only, so value(mask, depth) is the exact depth-limited
// minimax value whatever the move order. One instance per thread.
class BoxesAlphaBeta {
public:
    BoxesAlphaBeta(int rows, int cols = 0, std::size_t table_entries = std::size_t{1} << 20);

    int value(uint64_t mask, int depth);                       // side to move's remaining margin
    int best_edge(uint64_t mask, int depth, uint64_t seed);    // -1 only on a finished board
    int greedy_haul(uint64_t mask) const;                      // the depth-0 leaf score
    int classify(uint64_t mask, int edge) const;               // 0 capture, 1 safe, 2 loony
    uint64_t nodes() const { return nodes_; }
    std::size_t table_bytes() const { return table_.size() * sizeof(Entry); }

private:
    enum Flag : int8_t { kEmpty = 0, kExact = 1, kLower = 2, kUpper = 3 };
    struct Entry {
        uint64_t key;
        int16_t value;
        int8_t depth;
        int8_t flag;
    };

    int search(uint64_t mask, int depth, int alpha, int beta, std::mt19937_64& rng);
    int sides(uint64_t mask, int box) const;
    int captures(uint64_t mask, int edge) const;
    std::vector<int> ordered_moves(uint64_t mask, std::mt19937_64& rng) const;
    std::size_t index(uint64_t key, int depth) const {
        return static_cast<std::size_t>(((key ^ (uint64_t{0x9E37} * static_cast<uint64_t>(depth)))
                                         * 0x9E3779B97F4A7C15ULL) >> (64 - log2_size_));
    }

    BoxesBoard proto_;
    uint64_t full_;
    std::vector<Entry> table_;
    int log2_size_ = 0;
    uint64_t nodes_ = 0;
};

}  // namespace alpha_go
