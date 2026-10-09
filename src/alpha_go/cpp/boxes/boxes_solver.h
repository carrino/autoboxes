#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "boxes/boxes_chains.h"
#include "boxes/boxes_game.h"

namespace alpha_go {

// Exact endgame solver (PLAN.md §4.1), the C++ port of src/alpha_go/boxes/solver.py.
// value(mask) is the remaining box margin for the side to move under optimal play, a
// function of the edge mask alone. Negamax alpha-beta over the same four exact
// reductions as the reference: forced captures (chains::collapse_mask), a transposition
// table keyed on the symmetry-canonical mask (bounded, replace-always, so its byte size
// is a constructor argument), one representative edge per independent chain or loop,
// and the exact value of simple loony endgames by recursion over component sizes.
// value_within() stops after a node budget so the search can use it at every late leaf.
// One instance per thread: the table is not synchronised.
class BoxesSolver {
public:
    BoxesSolver(int rows, int cols = 0, std::size_t table_entries = std::size_t{1} << 20);

    int value(uint64_t mask);                                            // unbounded
    std::optional<int> value_within(uint64_t mask, uint64_t max_nodes);  // nullopt if over budget
    int remaining(const BoxesBoard& board) { return value(board.edges()); }
    int final_margin(const BoxesBoard& board) { return board.margin() + value(board.edges()); }
    int best_edge(const BoxesBoard& board);  // an optimal edge, forced captures first

    uint64_t canonical(uint64_t mask) const;
    int loony_value(std::vector<int> chains, std::vector<int> loops);

    std::size_t table_entries() const { return table_.size(); }
    std::size_t table_bytes() const { return table_.size() * sizeof(Entry); }
    uint64_t nodes() const { return nodes_; }  // nodes visited by the last call
    const BoxesGeometry& geometry() const { return proto_.geometry(); }

private:
    enum Flag : int8_t { kEmpty = 0, kExact = 1, kLower = 2, kUpper = 3 };
    struct Entry {
        uint64_t key;
        int16_t value;
        int8_t flag;
    };

    int search(uint64_t mask, int alpha, int beta);
    int child_after_take(uint64_t quiet, const chains::Decision& decision, int alpha, int beta);
    std::vector<int> moves(uint64_t quiet, const std::vector<int>& deg,
                           const std::vector<chains::Component>& comps) const;
    std::size_t index(uint64_t key) const {
        return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> (64 - log2_size_));
    }

    BoxesBoard proto_;  // keeps the shared geometry alive
    uint64_t full_;
    std::vector<std::vector<int>> perms_;  // edge permutation per lattice transform
    std::vector<Entry> table_;
    int log2_size_ = 0;
    uint64_t nodes_ = 0;
    uint64_t budget_ = 0;
    bool aborted_ = false;
    std::map<std::pair<std::vector<int>, std::vector<int>>, int> loony_cache_;
};

}  // namespace alpha_go
