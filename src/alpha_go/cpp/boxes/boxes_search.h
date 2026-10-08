#pragma once

#include <vector>

#include "boxes/boxes_game.h"

namespace alpha_go {

// Forced-move collapse (PLAN.md §2.5), mirroring src/alpha_go/boxes/forced.py:
// a position at a decision point. Auto-captures for the side to move are played
// on construction (recorded in prefix()); if the last opened component leaves a
// take-all / keep-control choice, the two macro-actions are exposed by their
// first edge, otherwise the plain undrawn edges are the actions. Satisfies the
// MCTSTree State concept and the evaluator surface of BoxesBoard.
class BoxesSearchState {
public:
    static constexpr int GROUND = -1;
    static constexpr int OPEN = -2;

    explicit BoxesSearchState(const BoxesBoard& board);

    // MCTS State concept
    bool is_game_over() const { return board_.is_game_over(); }
    int player() const { return board_.player(); }
    void apply(int action);
    float outcome(int player) const { return board_.outcome(player); }

    // Macro-actions by first edge
    std::vector<int> get_legal_moves_flat() const;
    const std::vector<int>& prefix() const { return prefix_; }
    bool has_decision() const { return has_decision_; }
    const std::vector<int>& decision_take() const { return take_; }
    int decision_control() const { return control_; }

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
    struct Component {
        std::vector<int> boxes;
        bool is_loop;
        int end0;
        int end1;
        bool opened() const { return end0 == OPEN || end1 == OPEN; }
    };

    void collapse();
    std::vector<Component> components(uint64_t mask, const std::vector<int>& deg) const;
    std::vector<int> take_sequence(uint64_t mask, const std::vector<int>& boxes) const;
    int control_edge(uint64_t mask, const Component& comp, const std::vector<int>& deg) const;
    std::vector<int> open_end_first(const Component& comp, const std::vector<int>& deg) const;

    BoxesBoard board_;
    std::vector<int> prefix_;
    bool has_decision_ = false;
    std::vector<int> take_;
    int control_ = -1;
};

}  // namespace alpha_go
