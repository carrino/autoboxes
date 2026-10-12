#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace alpha_go {

// Dots and Boxes on R x C boxes. Mirrors the Python reference in
// src/alpha_go/boxes/rules.py: actions are edge indices (horizontal edges
// r*C + c first, then vertical edges (R+1)*C + r*(C+1) + c), every edge also
// has lattice coordinates on the (2R+1) x (2C+1) grid, completing the fourth
// side of a box captures it for the mover who then moves again, and the game
// ends when every edge is drawn. Player codes mirror Go's BLACK/WHITE so the
// shared play loop can key agents by to_play().

// Index tables for one board size, shared by every copy of a board.
struct BoxesGeometry {
    int rows;
    int cols;
    int num_edges;
    int num_boxes;
    std::vector<std::array<int, 4>> box_edges;    // box -> top, bottom, left, right
    std::vector<std::array<int, 2>> edge_boxes;   // edge -> adjacent boxes, -1 = none
    std::vector<std::pair<int, int>> edge_rc;     // edge -> lattice (row, col)
    std::vector<int> lattice_edge;                // lattice row-major -> edge or -1

    BoxesGeometry(int rows, int cols);
    int lattice_rows() const { return 2 * rows + 1; }
    int lattice_cols() const { return 2 * cols + 1; }
};

class BoxesBoard {
public:
    static constexpr int8_t PLAYER_1 = 1;  // moves first
    static constexpr int8_t PLAYER_2 = 2;
    static constexpr int MAX_EDGES = 64;   // edge mask is one uint64_t
    static constexpr int MAX_BOXES = 32;   // any board with <= 64 edges has <= 25 boxes

    // cols <= 0 means a square board. Throws std::invalid_argument above MAX_EDGES.
    explicit BoxesBoard(int rows, int cols = 0);
    BoxesBoard(const BoxesBoard&) = default;
    BoxesBoard& operator=(const BoxesBoard&) = default;

    // Geometry
    int rows() const { return geo_->rows; }
    int cols() const { return geo_->cols; }
    int num_edges() const { return geo_->num_edges; }
    int num_boxes() const { return geo_->num_boxes; }
    const BoxesGeometry& geometry() const { return *geo_; }
    std::pair<int, int> row_col(int edge) const { return geo_->edge_rc[edge]; }
    int edge_index(int row, int col) const;  // -1 if (row, col) is not an edge cell

    // State
    uint64_t edges() const { return edges_; }
    int8_t to_play() const { return to_play_; }
    int player() const { return to_play_ - 1; }  // 0 or 1, for the MCTS State concept
    int move_count() const { return move_count_; }
    int boxes(int player) const { return boxes_[player]; }
    int8_t owner(int box) const { return owner_[box]; }
    int sides(int box) const { return sides_[box]; }

    // Rules
    bool is_legal_edge(int edge) const;
    bool is_legal(int row, int col) const { return is_legal_edge(edge_index(row, col)); }
    std::vector<int> get_legal_moves_flat() const;
    bool play_edge(int edge);  // false if already drawn; captures keep the turn
    bool play(int row, int col) { return play_edge(edge_index(row, col)); }
    void apply(int action) { play_edge(action); }  // MCTS State concept
    bool is_game_over() const { return edges_ == full_; }
    float score() const { return static_cast<float>(boxes_[0] - boxes_[1]); }
    int margin() const { return boxes_[player()] - boxes_[1 - player()]; }
    int8_t get_winner() const;         // PLAYER_1, PLAYER_2, or 0 for a tie
    float outcome(int player) const;   // 1 / 0.5 / 0 for the MCTS State concept

    // Export: lattice grid, drawn edges 1, captured boxes their owner's code
    std::vector<int8_t> to_lattice() const;
    // Feature planes for the net, float32 (kNumPlanes, lattice_rows, lattice_cols) row-major,
    // identical to alpha_go.boxes.encode.encode_grid (see that docstring for the plane list).
    static constexpr int kNumPlanes = 11;
    // With `chains`, the 10 planes of the "chains" feature set follow (chain / loop
    // structure from boxes_chains; see encode.py for the plane list).
    static constexpr int kNumChainPlanes = 10;
    void encode_planes(float* out, bool chains = false) const;
    std::string render() const;

private:
    std::shared_ptr<const BoxesGeometry> geo_;
    uint64_t full_;
    uint64_t edges_ = 0;
    std::array<int8_t, MAX_BOXES> sides_{};
    std::array<int8_t, MAX_BOXES> owner_{};
    std::array<int, 2> boxes_{};
    int8_t to_play_ = PLAYER_1;
    int move_count_ = 0;
};

// Count move sequences of length `depth` (shorter if the game ends): returns
// (sequences, boxes of PLAYER_1 summed over leaves, same for PLAYER_2).
std::tuple<long long, long long, long long> boxes_perft(const BoxesBoard& board, int depth);

}  // namespace alpha_go
