#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <algorithm>

#include "boxes/boxes_game.h"
#include "boxes/boxes_search.h"
#include "boxes/boxes_solver.h"
#include "go/go_game.h"
#include "mcts/mcts.h"

namespace py = pybind11;

// Bind MCTSTree<State> under `name`. Evaluators are Python callables taking the
// state (or a list of states for the batched path) and returning
// (dict[action, prob], value) where value is the side-to-move win probability.
template <class State>
void bind_mcts_tree(py::module_& m, const char* name) {
    using Tree = alpha_go::MCTSTree<State>;
    py::class_<Tree>(m, name)
        .def(py::init<const State&, const alpha_go::MCTSConfig&>(),
             py::arg("root_state"), py::arg("config"),
             "Create MCTS tree from root state with given config.")
        .def("run_simulations", &Tree::run_simulations,
             py::arg("num_simulations"), py::arg("evaluator"),
             "Run MCTS simulations using the evaluator function.\n"
             "evaluator: callable(state) -> (dict[int, float], float)\n"
             "Returns (action -> probability dict, value estimate).")
        .def("get_action_probabilities", &Tree::get_action_probabilities,
             py::arg("temperature") = 1.0f,
             "Get action probabilities based on visit counts.\n"
             "temperature=0 gives deterministic (argmax), temperature=1 proportional.")
        .def("select_action", &Tree::select_action,
             py::arg("temperature") = 1.0f,
             "Select an action based on visit counts and temperature.")
        .def("tree_size", &Tree::tree_size,
             "Get number of nodes in the tree.")
        .def("get_root_visit_count", &Tree::get_root_visit_count,
             "Get visit count of root node.")
        .def("get_root_q_value", &Tree::get_root_q_value,
             "Get Q-value of root node (player_at_parent / opponent perspective).")
        .def("get_root_policy_priors", &Tree::get_root_policy_priors,
             "Get root policy priors as dict[action, probability]. Includes Dirichlet noise if applied.")
        .def("get_child_visit_counts", &Tree::get_child_visit_counts,
             "Get visit counts of root's children as dict[action, count].")
        .def("get_child_q_values", &Tree::get_child_q_values,
             "Get Q-values of root's children as dict[action, q_value].")
        .def("get_child_first_eval_values", &Tree::get_child_first_eval_values,
             "Get the raw NN v_theta recorded at each root-child at expansion "
             "time as dict[action, value]. Same perspective as Q (root player).")
        .def("get_child_max_subtree_depths", &Tree::get_child_max_subtree_depths,
             "Get max subtree depth under each root child as dict[action, depth].")
        .def("run_simulations_batched", &Tree::run_simulations_batched,
             py::arg("num_simulations"), py::arg("leaf_batch_size"), py::arg("batched_evaluator"),
             "Leaf-parallel MCTS with virtual loss.\n"
             "batched_evaluator: callable(list[state]) -> list[(dict[int,float], float)]");
}

PYBIND11_MODULE(alpha_go_cpp, m) {
    m.doc() = "C++ backend for AlphaGo MCTS and Go game";

    // Constants
    m.attr("PASS_ACTION") = alpha_go::PASS_ACTION;

    // GoBoard binding
    py::class_<alpha_go::GoBoard>(m, "GoBoard")
        .def(py::init<int, float>(), py::arg("size") = 9, py::arg("komi") = alpha_go::GoBoard::KOMI)
        .def("play", &alpha_go::GoBoard::play, py::arg("row"), py::arg("col"),
             "Play a stone at (row, col). Returns true if legal.")
        .def("play_flat", &alpha_go::GoBoard::play_flat, py::arg("index"),
             "Play a stone at flat index. Returns true if legal.")
        .def("pass_move", &alpha_go::GoBoard::pass,
             "Pass the turn. Returns true.")
        .def("is_legal", &alpha_go::GoBoard::is_legal, py::arg("row"), py::arg("col"),
             "Check if playing at (row, col) is legal.")
        .def("is_legal_flat", &alpha_go::GoBoard::is_legal_flat, py::arg("index"),
             "Check if playing at flat index is legal.")
        .def("get_legal_moves_flat", &alpha_go::GoBoard::get_legal_moves_flat,
             "Get list of legal move indices (not including pass).")
        .def("is_game_over", &alpha_go::GoBoard::is_game_over,
             "Check if game is over (two consecutive passes).")
        .def("score", &alpha_go::GoBoard::score,
             "Get score (black - white) using Chinese rules.")
        .def("get_winner", &alpha_go::GoBoard::get_winner,
             "Get winner: BLACK (1), WHITE (2), or 0 for draw.")
        .def("size", &alpha_go::GoBoard::size,
             "Get board size.")
        .def("to_play", &alpha_go::GoBoard::to_play,
             "Get current player: BLACK (1) or WHITE (2).")
        .def("move_count", &alpha_go::GoBoard::move_count,
             "Get total number of moves played.")
        .def("komi", &alpha_go::GoBoard::komi,
             "Get komi value for this board.")
        .def("at", &alpha_go::GoBoard::at, py::arg("row"), py::arg("col"),
             "Get stone at (row, col): EMPTY (0), BLACK (1), WHITE (2).")
        .def("row_col", &alpha_go::GoBoard::row_col, py::arg("flat_index"),
             "Convert flat index to (row, col) pair.")
        .def("copy", [](const alpha_go::GoBoard& b) { return alpha_go::GoBoard(b); },
             "Create a copy of the board.")
        .def("to_numpy", [](const alpha_go::GoBoard& b) {
            // Return numpy array copy of board
            auto arr = py::array_t<int8_t>({b.size(), b.size()});
            auto buf = arr.mutable_unchecked<2>();
            for (int i = 0; i < b.size(); i++) {
                for (int j = 0; j < b.size(); j++) {
                    buf(i, j) = b.at(i, j);
                }
            }
            return arr;
        }, "Convert board to numpy array.")
        .def("set_from_numpy", [](alpha_go::GoBoard& b, py::array_t<int8_t> arr, int8_t to_play) {
            // Set board state from numpy array
            auto buf = arr.unchecked<2>();
            if (buf.shape(0) != b.size() || buf.shape(1) != b.size()) {
                throw std::runtime_error("Array shape must match board size");
            }
            // Flatten and copy
            std::vector<int8_t> flat(b.size() * b.size());
            for (int i = 0; i < b.size(); i++) {
                for (int j = 0; j < b.size(); j++) {
                    flat[i * b.size() + j] = buf(i, j);
                }
            }
            b.set_from_array(flat.data(), to_play);
        }, py::arg("board_array"), py::arg("to_play"),
           "Set board state from numpy array and current player.")
        .def("__repr__", [](const alpha_go::GoBoard& b) {
            std::string s = "GoBoard(" + std::to_string(b.size()) + "x" +
                           std::to_string(b.size()) + ", to_play=";
            s += (b.to_play() == alpha_go::GoBoard::BLACK) ? "BLACK" : "WHITE";
            s += ", moves=" + std::to_string(b.move_count()) + ")";
            return s;
        })
        .def_readonly_static("EMPTY", &alpha_go::GoBoard::EMPTY)
        .def_readonly_static("BLACK", &alpha_go::GoBoard::BLACK)
        .def_readonly_static("WHITE", &alpha_go::GoBoard::WHITE)
        .def_readonly_static("KOMI", &alpha_go::GoBoard::KOMI);

    // BoxesBoard binding (Dots and Boxes)
    py::class_<alpha_go::BoxesBoard>(m, "BoxesBoard")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols") = 0,
             "Create an empty R x C Boxes board (cols=0 means square).")
        .def("rows", &alpha_go::BoxesBoard::rows)
        .def("cols", &alpha_go::BoxesBoard::cols)
        .def("num_edges", &alpha_go::BoxesBoard::num_edges)
        .def("num_boxes", &alpha_go::BoxesBoard::num_boxes)
        .def("to_play", &alpha_go::BoxesBoard::to_play,
             "Side to move: PLAYER_1 (1) or PLAYER_2 (2).")
        .def("player", &alpha_go::BoxesBoard::player, "Side to move as 0 / 1.")
        .def("move_count", &alpha_go::BoxesBoard::move_count)
        .def("boxes", &alpha_go::BoxesBoard::boxes, py::arg("player"),
             "Boxes captured by player index 0 / 1.")
        .def("owner", &alpha_go::BoxesBoard::owner, py::arg("box"))
        .def("sides", &alpha_go::BoxesBoard::sides, py::arg("box"))
        .def("edges", &alpha_go::BoxesBoard::edges, "Drawn-edge bitmask.")
        .def("row_col", &alpha_go::BoxesBoard::row_col, py::arg("edge"),
             "Lattice (row, col) of an edge.")
        .def("edge_index", &alpha_go::BoxesBoard::edge_index, py::arg("row"), py::arg("col"),
             "Edge at lattice (row, col), or -1.")
        .def("is_legal_edge", &alpha_go::BoxesBoard::is_legal_edge, py::arg("edge"))
        .def("is_legal", &alpha_go::BoxesBoard::is_legal, py::arg("row"), py::arg("col"))
        .def("get_legal_moves_flat", &alpha_go::BoxesBoard::get_legal_moves_flat,
             "Undrawn edge indices.")
        .def("play_edge", &alpha_go::BoxesBoard::play_edge, py::arg("edge"),
             "Draw an edge. Returns true if legal. Captures keep the turn.")
        .def("play", &alpha_go::BoxesBoard::play, py::arg("row"), py::arg("col"),
             "Draw the edge at lattice (row, col). Returns true if legal.")
        .def("is_game_over", &alpha_go::BoxesBoard::is_game_over)
        .def("score", &alpha_go::BoxesBoard::score, "boxes(PLAYER_1) - boxes(PLAYER_2).")
        .def("margin", &alpha_go::BoxesBoard::margin, "Box difference for the side to move.")
        .def("get_winner", &alpha_go::BoxesBoard::get_winner,
             "PLAYER_1 (1), PLAYER_2 (2), or 0 for a tie.")
        .def("outcome", &alpha_go::BoxesBoard::outcome, py::arg("player"),
             "1 / 0.5 / 0 for player index 0 / 1.")
        .def("copy", [](const alpha_go::BoxesBoard& b) { return alpha_go::BoxesBoard(b); })
        .def("to_numpy", [](const alpha_go::BoxesBoard& b) {
            const auto& geo = b.geometry();
            auto arr = py::array_t<int8_t>({geo.lattice_rows(), geo.lattice_cols()});
            std::vector<int8_t> grid = b.to_lattice();
            std::copy(grid.begin(), grid.end(), arr.mutable_data());
            return arr;
        }, "Lattice grid: drawn edges 1, captured boxes their owner's code.")
        .def("render", &alpha_go::BoxesBoard::render, "ASCII picture.")
        .def("__repr__", [](const alpha_go::BoxesBoard& b) {
            return "BoxesBoard(" + std::to_string(b.rows()) + "x" + std::to_string(b.cols()) +
                   ", to_play=" + std::to_string(b.to_play()) +
                   ", moves=" + std::to_string(b.move_count()) + ")";
        })
        .def_readonly_static("PLAYER_1", &alpha_go::BoxesBoard::PLAYER_1)
        .def_readonly_static("PLAYER_2", &alpha_go::BoxesBoard::PLAYER_2);

    m.def("boxes_perft", &alpha_go::boxes_perft, py::arg("board"), py::arg("depth"),
          "(sequences, PLAYER_1 boxes summed over leaves, PLAYER_2 boxes summed over leaves).");

    m.def("encode_planes", [](const py::sequence& states) {
        // Feature planes (B, 11, H, W) for a batch of BoxesBoard / BoxesSearchState of one size.
        const auto board_of = [](py::handle h) -> const alpha_go::BoxesBoard& {
            if (py::isinstance<alpha_go::BoxesSearchState>(h)) {
                return h.cast<const alpha_go::BoxesSearchState&>().board();
            }
            return h.cast<const alpha_go::BoxesBoard&>();
        };
        const py::ssize_t n = py::len(states);
        if (n == 0) {
            throw std::invalid_argument("encode_planes: empty batch");
        }
        const auto& geo = board_of(states[0]).geometry();
        auto arr = py::array_t<float>({n, static_cast<py::ssize_t>(alpha_go::BoxesBoard::kNumPlanes),
                                       static_cast<py::ssize_t>(geo.lattice_rows()),
                                       static_cast<py::ssize_t>(geo.lattice_cols())});
        const py::ssize_t stride = alpha_go::BoxesBoard::kNumPlanes * geo.lattice_rows() * geo.lattice_cols();
        float* out = arr.mutable_data();
        for (py::ssize_t i = 0; i < n; ++i) {
            const alpha_go::BoxesBoard& board = board_of(states[i]);
            if (board.rows() != geo.rows || board.cols() != geo.cols) {
                throw std::invalid_argument("encode_planes: boards of different sizes");
            }
            board.encode_planes(out + i * stride);
        }
        return arr;
    }, py::arg("states"),
    "Feature planes (B, 11, H, W) float32 for BoxesBoard / BoxesSearchState objects of one size; "
    "identical to alpha_go.boxes.encode.encode_batch.");

    // BoxesSolver binding: exact endgame solver (PLAN.md §4.1)
    py::class_<alpha_go::BoxesSolver, std::shared_ptr<alpha_go::BoxesSolver>>(m, "BoxesSolver")
        .def(py::init<int, int, std::size_t>(), py::arg("rows"), py::arg("cols") = 0,
             py::arg("table_entries") = std::size_t{1} << 20,
             "Exact remaining-margin solver with a bounded transposition table (one per thread).")
        .def("value", &alpha_go::BoxesSolver::value, py::arg("mask"),
             "Remaining box margin for the side to move under optimal play (unbounded search).")
        .def("value_within", &alpha_go::BoxesSolver::value_within, py::arg("mask"),
             py::arg("max_nodes"), "Like value(), or None once max_nodes nodes were visited.")
        .def("remaining", &alpha_go::BoxesSolver::remaining, py::arg("board"))
        .def("final_margin", &alpha_go::BoxesSolver::final_margin, py::arg("board"),
             "Final margin for the side to move under optimal play from `board`.")
        .def("best_edge", &alpha_go::BoxesSolver::best_edge, py::arg("board"),
             "An optimal edge for the side to move (forced captures first).")
        .def("canonical", &alpha_go::BoxesSolver::canonical, py::arg("mask"))
        .def("loony_value", &alpha_go::BoxesSolver::loony_value, py::arg("chains"), py::arg("loops"),
             "Exact value of a simple loony endgame from its chain and loop sizes.")
        .def("table_entries", &alpha_go::BoxesSolver::table_entries)
        .def("table_bytes", &alpha_go::BoxesSolver::table_bytes)
        .def("nodes", &alpha_go::BoxesSolver::nodes, "Nodes visited by the last call.");

    // BoxesSearchState binding: forced-move collapse around a BoxesBoard
    py::class_<alpha_go::BoxesSearchState>(m, "BoxesSearchState")
        .def(py::init<const alpha_go::BoxesBoard&>(), py::arg("board"),
             "Collapse the side to move's forced captures; see get_legal_moves_flat().")
        .def(py::init<const alpha_go::BoxesBoard&, std::shared_ptr<alpha_go::BoxesSolver>, int,
                      uint64_t>(),
             py::arg("board"), py::arg("solver"), py::arg("max_undrawn"), py::arg("node_budget"),
             "As above, and positions with <= max_undrawn undrawn edges that the solver settles "
             "within node_budget nodes are terminal with the exact outcome.")
        .def("solved", &alpha_go::BoxesSearchState::solved)
        .def("solved_margin", &alpha_go::BoxesSearchState::solved_margin,
             "Exact final margin for the side to move when solved().")
        .def("get_legal_moves_flat", &alpha_go::BoxesSearchState::get_legal_moves_flat,
             "Macro-actions by first edge (take-all / keep-control at a decision, else edges).")
        .def("prefix", &alpha_go::BoxesSearchState::prefix,
             "Edges auto-played from the constructing position (all by the same mover).")
        .def("has_decision", &alpha_go::BoxesSearchState::has_decision)
        .def("decision_take", &alpha_go::BoxesSearchState::decision_take)
        .def("decision_control", &alpha_go::BoxesSearchState::decision_control)
        .def("apply", &alpha_go::BoxesSearchState::apply, py::arg("action"),
             "Play a macro-action and collapse for the new side to move.")
        .def("board", &alpha_go::BoxesSearchState::board,
             py::return_value_policy::copy, "Copy of the underlying board.")
        .def("to_play", &alpha_go::BoxesSearchState::to_play)
        .def("player", &alpha_go::BoxesSearchState::player)
        .def("rows", &alpha_go::BoxesSearchState::rows)
        .def("cols", &alpha_go::BoxesSearchState::cols)
        .def("num_edges", &alpha_go::BoxesSearchState::num_edges)
        .def("edges", &alpha_go::BoxesSearchState::edges)
        .def("margin", &alpha_go::BoxesSearchState::margin)
        .def("move_count", &alpha_go::BoxesSearchState::move_count)
        .def("boxes", &alpha_go::BoxesSearchState::boxes, py::arg("player"))
        .def("is_game_over", &alpha_go::BoxesSearchState::is_game_over)
        .def("outcome", &alpha_go::BoxesSearchState::outcome, py::arg("player"))
        .def("row_col", &alpha_go::BoxesSearchState::row_col, py::arg("edge"))
        .def("copy", [](const alpha_go::BoxesSearchState& s) { return alpha_go::BoxesSearchState(s); })
        .def("to_numpy", [](const alpha_go::BoxesSearchState& s) {
            const auto& geo = s.board().geometry();
            auto arr = py::array_t<int8_t>({geo.lattice_rows(), geo.lattice_cols()});
            std::vector<int8_t> grid = s.to_lattice();
            std::copy(grid.begin(), grid.end(), arr.mutable_data());
            return arr;
        })
        .def("render", &alpha_go::BoxesSearchState::render);

    // MCTSConfig binding
    py::class_<alpha_go::MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("c_puct", &alpha_go::MCTSConfig::c_puct,
                      "PUCT exploration constant (default: 1.0)")
        .def_readwrite("lambda_", &alpha_go::MCTSConfig::lambda_,
                      "Mix between value and rollout (0 = pure value, default: 0.0)")
        .def_readwrite("dirichlet_alpha", &alpha_go::MCTSConfig::dirichlet_alpha,
                      "Dirichlet noise alpha (0 = no noise, default: 0.0)")
        .def_readwrite("dirichlet_weight", &alpha_go::MCTSConfig::dirichlet_weight,
                      "Weight of Dirichlet noise (default: 0.25)")
        .def_readwrite("temperature", &alpha_go::MCTSConfig::temperature,
                      "Temperature for action selection (default: 1.0)")
        .def_readwrite("max_depth", &alpha_go::MCTSConfig::max_depth,
                      "Maximum total depth from game start (tree + rollout combined, default: 100)")
        .def_readwrite("rollout_temperature", &alpha_go::MCTSConfig::rollout_temperature,
                      "Temperature for sampling during fast rollouts (default: 1.0)")
        .def_readwrite("pcr_sims", &alpha_go::MCTSConfig::pcr_sims,
                      "Playout cap randomization: list of sim counts to sample from.")
        .def_readwrite("pcr_probs", &alpha_go::MCTSConfig::pcr_probs,
                      "Playout cap randomization: categorical probabilities (must match pcr_sims length, sum to 1).")
        .def("__repr__", [](const alpha_go::MCTSConfig& c) {
            return "MCTSConfig(c_puct=" + std::to_string(c.c_puct) +
                   ", lambda=" + std::to_string(c.lambda_) +
                   ", dirichlet_alpha=" + std::to_string(c.dirichlet_alpha) + ")";
        });

    // MCTSTree bindings: the same template for Go ("MCTSTree") and Boxes ("BoxesMCTSTree")
    bind_mcts_tree<alpha_go::GoBoard>(m, "MCTSTree");
    bind_mcts_tree<alpha_go::BoxesBoard>(m, "BoxesMCTSTree");
    bind_mcts_tree<alpha_go::BoxesSearchState>(m, "BoxesSearchMCTSTree");

    // Convenience function for running MCTS with a Python evaluator
    m.def("run_mcts", [](
        const alpha_go::GoBoard& state,
        int num_simulations,
        const alpha_go::MCTSConfig& config,
        py::function evaluator,
        float temperature
    ) {
        alpha_go::GoMCTSTree tree(state, config);

        // Wrap Python evaluator
        auto cpp_evaluator = [&evaluator](const alpha_go::GoBoard& s)
            -> std::pair<std::unordered_map<int, float>, float> {
            py::object result = evaluator(s);
            auto policy = result.attr("__getitem__")(0).cast<std::unordered_map<int, float>>();
            auto value = result.attr("__getitem__")(1).cast<float>();
            return std::make_pair(policy, value);
        };

        tree.run_simulations(num_simulations, cpp_evaluator);
        return tree.get_action_probabilities(temperature);
    },
    py::arg("state"),
    py::arg("num_simulations"),
    py::arg("config"),
    py::arg("evaluator"),
    py::arg("temperature") = 1.0f,
    "Run MCTS search and return action probabilities.\n"
    "state: GoBoard root state\n"
    "num_simulations: number of MCTS simulations\n"
    "config: MCTSConfig\n"
    "evaluator: callable(GoBoard) -> (dict[int, float], float)\n"
    "temperature: temperature for action selection\n"
    "Returns: dict[action, probability]");

    // Version info
    m.attr("__version__") = "0.1.0";
}
