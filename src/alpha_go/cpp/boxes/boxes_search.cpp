#include "boxes_search.h"

namespace alpha_go {

BoxesSearchState::BoxesSearchState(const BoxesBoard& board) : board_(board) {
    collapse();
}

BoxesSearchState::BoxesSearchState(const BoxesBoard& board, std::shared_ptr<BoxesSolver> solver,
                                   int max_undrawn, uint64_t node_budget)
    : board_(board), solver_(std::move(solver)), max_undrawn_(max_undrawn),
      node_budget_(node_budget) {
    collapse();
}

std::vector<int> BoxesSearchState::get_legal_moves_flat() const {
    if (has_decision_) {
        return {take_[0], control_};
    }
    return board_.get_legal_moves_flat();
}

void BoxesSearchState::apply(int action) {
    if (has_decision_ && action == take_[0]) {
        for (int e : take_) {
            board_.play_edge(e);
        }
    } else {
        board_.play_edge(action);
    }
    prefix_.clear();
    has_decision_ = false;
    take_.clear();
    control_ = -1;
    collapse();
}

float BoxesSearchState::outcome(int player) const {
    if (!solved_) {
        return board_.outcome(player);
    }
    const int margin = player == board_.player() ? solved_margin_ : -solved_margin_;
    return margin > 0 ? 1.0f : (margin == 0 ? 0.5f : 0.0f);
}

void BoxesSearchState::collapse() {
    chains::Decision decision;
    std::vector<int> prefix;
    chains::collapse_mask(board_.edges(), board_.geometry(), prefix, decision);
    for (int e : prefix) {
        board_.play_edge(e);
        prefix_.push_back(e);
    }
    if (decision.control >= 0) {
        has_decision_ = true;
        take_ = decision.take;
        control_ = decision.control;
    }
    solve();
}

void BoxesSearchState::solve() {
    solved_ = false;
    if (!solver_ || board_.is_game_over() ||
        board_.num_edges() - board_.move_count() > max_undrawn_) {
        return;
    }
    const std::optional<int> remaining = solver_->value_within(board_.edges(), node_budget_);
    if (remaining) {
        solved_ = true;
        solved_margin_ = board_.margin() + *remaining;
    }
}

}  // namespace alpha_go
