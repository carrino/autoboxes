#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "boxes/boxes_chains.h"
#include "boxes/boxes_game.h"
#include "boxes/boxes_solver.h"

namespace alpha_go {

// Forced-move collapse (PLAN.md §2.5), mirroring src/alpha_go/boxes/forced.py:
// a position at a decision point. Auto-captures for the side to move are played
// on construction (recorded in prefix()); if the last opened component leaves a
// take-all / keep-control choice, the two macro-actions are exposed by their
// first edge, otherwise the plain undrawn edges are the actions. Satisfies the
// MCTSTree State concept and the evaluator surface of BoxesBoard.
//
// With a solver (PLAN.md §4.1) a position with at most max_undrawn undrawn edges
// that the solver settles within node_budget nodes is terminal for the search:
// is_game_over() is true and outcome() is the exact 1 / 0.5 / 0 from the final
// margin, so the net is never asked about it. Children inherit the solver.
class BoxesSearchState {
public:
    static constexpr int GROUND = chains::GROUND;
    static constexpr int OPEN = chains::OPEN;

    explicit BoxesSearchState(const BoxesBoard& board);
    BoxesSearchState(const BoxesBoard& board, std::shared_ptr<BoxesSolver> solver,
                     int max_undrawn, uint64_t node_budget);

    // MCTS State concept
    bool is_game_over() const { return solved_ || board_.is_game_over(); }
    int player() const { return board_.player(); }
    void apply(int action);
    float outcome(int player) const;

    // Macro-actions by first edge
    std::vector<int> get_legal_moves_flat() const;
    const std::vector<int>& prefix() const { return prefix_; }
    bool has_decision() const { return has_decision_; }
    const std::vector<int>& decision_take() const { return take_; }
    int decision_control() const { return control_; }

    // Solver verdict (see class comment)
    bool solved() const { return solved_; }
    int solved_margin() const { return solved_margin_; }  // final margin, side to move

    // Board surface used by the evaluators and the play loop
    const BoxesBoard& board() const { return board_; }
    int8_t to_play() const { return board_.to_play(); }
    int rows() const { return board_.rows(); }
    int cols() const { return board_.cols(); }
    int num_edges() const { return board_.num_edges(); }
    uint64_t edges() const { return board_.edges(); }
    int margin() const { return board_.margin(); }
    int move_count() const { return board_.move_count(); }
    int boxes(int player) const { return board_.boxes(player); }
    std::pair<int, int> row_col(int edge) const { return board_.row_col(edge); }
    std::vector<int8_t> to_lattice() const { return board_.to_lattice(); }
    std::string render() const { return board_.render(); }

private:
    void collapse();
    void solve();

    BoxesBoard board_;
    std::vector<int> prefix_;
    bool has_decision_ = false;
    std::vector<int> take_;
    int control_ = -1;
    std::shared_ptr<BoxesSolver> solver_;
    int max_undrawn_ = 0;
    uint64_t node_budget_ = 0;
    bool solved_ = false;
    int solved_margin_ = 0;
};

}  // namespace alpha_go
