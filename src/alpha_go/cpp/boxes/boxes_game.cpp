#include "boxes_game.h"

#include "boxes_chains.h"

#include <algorithm>

#include <stdexcept>

namespace alpha_go {

BoxesGeometry::BoxesGeometry(int rows_, int cols_)
    : rows(rows_),
      cols(cols_),
      num_edges((rows_ + 1) * cols_ + rows_ * (cols_ + 1)),
      num_boxes(rows_ * cols_),
      box_edges(num_boxes),
      edge_boxes(num_edges, std::array<int, 2>{-1, -1}),
      edge_rc(num_edges),
      lattice_edge((2 * rows_ + 1) * (2 * cols_ + 1), -1) {
    auto h = [&](int r, int c) { return r * cols + c; };
    auto v = [&](int r, int c) { return (rows + 1) * cols + r * (cols + 1) + c; };
    for (int r = 0; r <= rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            edge_rc[h(r, c)] = {2 * r, 2 * c + 1};
        }
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c <= cols; ++c) {
            edge_rc[v(r, c)] = {2 * r + 1, 2 * c};
        }
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int b = r * cols + c;
            box_edges[b] = {h(r, c), h(r + 1, c), v(r, c), v(r, c + 1)};
            for (int e : box_edges[b]) {
                edge_boxes[e][edge_boxes[e][0] < 0 ? 0 : 1] = b;
            }
        }
    }
    for (int e = 0; e < num_edges; ++e) {
        lattice_edge[edge_rc[e].first * lattice_cols() + edge_rc[e].second] = e;
    }
}

BoxesBoard::BoxesBoard(int rows, int cols)
    : geo_(std::make_shared<const BoxesGeometry>(rows, cols <= 0 ? rows : cols)) {
    if (geo_->num_edges > MAX_EDGES) {
        throw std::invalid_argument("BoxesBoard supports at most 64 edges");
    }
    full_ = (geo_->num_edges == 64) ? ~uint64_t{0} : (uint64_t{1} << geo_->num_edges) - 1;
}

int BoxesBoard::edge_index(int row, int col) const {
    if (row < 0 || row >= geo_->lattice_rows() || col < 0 || col >= geo_->lattice_cols()) {
        return -1;
    }
    return geo_->lattice_edge[row * geo_->lattice_cols() + col];
}

bool BoxesBoard::is_legal_edge(int edge) const {
    return edge >= 0 && edge < geo_->num_edges && !((edges_ >> edge) & 1);
}

std::vector<int> BoxesBoard::get_legal_moves_flat() const {
    std::vector<int> moves;
    moves.reserve(geo_->num_edges - move_count_);
    for (int e = 0; e < geo_->num_edges; ++e) {
        if (!((edges_ >> e) & 1)) {
            moves.push_back(e);
        }
    }
    return moves;
}

bool BoxesBoard::play_edge(int edge) {
    if (!is_legal_edge(edge)) {
        return false;
    }
    edges_ |= uint64_t{1} << edge;
    int captured = 0;
    for (int b : geo_->edge_boxes[edge]) {
        if (b >= 0 && ++sides_[b] == 4) {
            owner_[b] = to_play_;
            ++captured;
        }
    }
    boxes_[player()] += captured;
    ++move_count_;
    if (captured == 0) {
        to_play_ = (to_play_ == PLAYER_1) ? PLAYER_2 : PLAYER_1;
    }
    return true;
}

int8_t BoxesBoard::get_winner() const {
    int diff = boxes_[0] - boxes_[1];
    return diff > 0 ? PLAYER_1 : diff < 0 ? PLAYER_2 : 0;
}

float BoxesBoard::outcome(int player) const {
    int diff = boxes_[player] - boxes_[1 - player];
    return diff > 0 ? 1.0f : diff < 0 ? 0.0f : 0.5f;
}

std::vector<int8_t> BoxesBoard::to_lattice() const {
    const int lc = geo_->lattice_cols();
    std::vector<int8_t> grid(geo_->lattice_rows() * lc, 0);
    for (int e = 0; e < geo_->num_edges; ++e) {
        grid[geo_->edge_rc[e].first * lc + geo_->edge_rc[e].second] = (edges_ >> e) & 1;
    }
    for (int b = 0; b < geo_->num_boxes; ++b) {
        grid[(2 * (b / geo_->cols) + 1) * lc + 2 * (b % geo_->cols) + 1] = owner_[b];
    }
    return grid;
}

void BoxesBoard::encode_planes(float* out, bool chains) const {
    const int lr = geo_->lattice_rows();
    const int lc = geo_->lattice_cols();
    const int plane = lr * lc;
    std::fill(out, out + (kNumPlanes + (chains ? kNumChainPlanes : 0)) * plane, 0.0f);
    for (int e = 0; e < geo_->num_edges; ++e) {
        const int cell = geo_->edge_rc[e].first * lc + geo_->edge_rc[e].second;
        out[cell] = static_cast<float>((edges_ >> e) & 1);  // plane 0: edge drawn
        out[8 * plane + cell] = 1.0f;                        // plane 8: is-edge-cell
    }
    const int8_t mover = to_play_;
    for (int b = 0; b < geo_->num_boxes; ++b) {
        const int cell = (2 * (b / geo_->cols) + 1) * lc + 2 * (b % geo_->cols) + 1;
        out[1 * plane + cell] = owner_[b] == mover ? 1.0f : 0.0f;            // mine
        out[2 * plane + cell] = owner_[b] == 3 - mover ? 1.0f : 0.0f;        // theirs
        out[(3 + sides_[b]) * plane + cell] = 1.0f;                           // side count one-hot
        out[9 * plane + cell] = 1.0f;                                         // is-box-cell
    }
    const float margin = static_cast<float>(this->margin()) / static_cast<float>(geo_->num_boxes);
    std::fill(out + 10 * plane, out + 11 * plane, margin);
    if (!chains) {
        return;
    }
    float* ext = out + kNumPlanes * plane;
    const std::vector<int> deg = chains::degrees(edges_, *geo_);
    const std::vector<chains::Component> comps = chains::components(edges_, *geo_, deg);
    int long_chains = 0;
    int loops = 0;
    for (const chains::Component& comp : comps) {
        const int kind = comp.is_loop ? 3 : std::min(comp.size(), 3) - 1;
        long_chains += !comp.is_loop && comp.size() >= 3;
        loops += comp.is_loop;
        for (int b : comp.boxes) {
            const int cell = (2 * (b / geo_->cols) + 1) * lc + 2 * (b % geo_->cols) + 1;
            ext[kind * plane + cell] = 1.0f;
            ext[4 * plane + cell] = comp.opened() ? 1.0f : 0.0f;
        }
    }
    int safe = 0;
    for (int e = 0; e < geo_->num_edges; ++e) {
        if ((edges_ >> e) & 1) {
            continue;
        }
        bool ok = true;
        for (int b : geo_->edge_boxes[e]) {
            ok = ok && (b < 0 || deg[b] >= 3);
        }
        if (ok) {
            ext[5 * plane + geo_->edge_rc[e].first * lc + geo_->edge_rc[e].second] = 1.0f;
            ++safe;
        }
    }
    // Ratios in double then rounded once, like numpy's float64 -> float32.
    std::fill(ext + 6 * plane, ext + 7 * plane, static_cast<float>(long_chains / 4.0));
    std::fill(ext + 7 * plane, ext + 8 * plane, static_cast<float>(loops / 4.0));
    std::fill(ext + 8 * plane, ext + 9 * plane,
              static_cast<float>(static_cast<double>(safe) / geo_->num_edges));
    std::fill(ext + 9 * plane, ext + 10 * plane, static_cast<float>(long_chains % 2));
}

std::string BoxesBoard::render() const {
    const int lr = geo_->lattice_rows();
    const int lc = geo_->lattice_cols();
    std::vector<int8_t> grid = to_lattice();
    std::string out;
    for (int r = 0; r < lr; ++r) {
        for (int c = 0; c < lc; ++c) {
            int8_t value = grid[r * lc + c];
            bool odd_r = r % 2 == 1;
            bool odd_c = c % 2 == 1;
            if (!odd_r && !odd_c) {
                out += '.';
            } else if (!odd_r && odd_c) {
                out += value ? '-' : ' ';
            } else if (odd_r && !odd_c) {
                out += value ? '|' : ' ';
            } else {
                out += value ? static_cast<char>('0' + value) : ' ';
            }
        }
        if (r + 1 < lr) {
            out += '\n';
        }
    }
    return out;
}

std::tuple<long long, long long, long long> boxes_perft(const BoxesBoard& board, int depth) {
    if (depth == 0 || board.is_game_over()) {
        return {1, board.boxes(0), board.boxes(1)};
    }
    long long sequences = 0, p1 = 0, p2 = 0;
    for (int e : board.get_legal_moves_flat()) {
        BoxesBoard child = board;
        child.play_edge(e);
        auto [s, a, b] = boxes_perft(child, depth - 1);
        sequences += s;
        p1 += a;
        p2 += b;
    }
    return {sequences, p1, p2};
}

}  // namespace alpha_go
