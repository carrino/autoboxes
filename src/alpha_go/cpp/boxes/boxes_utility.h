#pragma once

#include <cmath>
#include <type_traits>

#include "boxes/boxes_game.h"
#include "boxes/boxes_search.h"

namespace alpha_go {

// Go always keeps its original outcome and proof rules, even with these knobs set.
template <class State>
bool shapes_terminal_margin(float lambda) {
    return (std::is_same_v<State, BoxesBoard> || std::is_same_v<State, BoxesSearchState>)
        && lambda != 0.0f;
}

// Python reference: boxes.utility.exact_value. The solver's margin already includes
// captured boxes; natural terminals use the board's final score. Preserve extra turns.
template <class State>
float terminal_utility(const State& state, int player, float lambda, float k) {
    const float win = state.outcome(player);
    if constexpr (std::is_same_v<State, BoxesBoard> || std::is_same_v<State, BoxesSearchState>) {
        if (lambda != 0.0f) {
            int margin = state.margin();
            if constexpr (std::is_same_v<State, BoxesSearchState>) {
                if (state.solved()) margin = state.solved_margin();
            }
            if (player != state.player()) margin = -margin;
            return win + lambda * std::tanh(static_cast<float>(margin) / k);
        }
    }
    return win;
}

}  // namespace alpha_go
