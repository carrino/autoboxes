#include "boxes_solver.h"

#include <algorithm>
#include <climits>

namespace alpha_go {

namespace {

constexpr int kInf = 1000;

// Lattice transform k (symmetry.py): optional flip of the column axis for k >= 4, then
// k % 4 counter-clockwise quarter turns; (r, c) -> (W - 1 - c, r) per turn.
std::pair<int, int> transform_cell(int r, int c, int height, int width, int k) {
    if (k >= 4) {
        c = width - 1 - c;
    }
    for (int i = 0; i < k % 4; ++i) {
        int nr = width - 1 - c;
        int nc = r;
        r = nr;
        c = nc;
        std::swap(height, width);
    }
    return {r, c};
}

}  // namespace

BoxesSolver::BoxesSolver(int rows, int cols, std::size_t table_entries)
    : proto_(rows, cols), full_(chains::full_mask(proto_.geometry())) {
    const BoxesGeometry& geo = proto_.geometry();
    const int height = geo.lattice_rows();
    const int width = geo.lattice_cols();
    const bool square = geo.rows == geo.cols;
    for (int k = 0; k < 8; ++k) {
        if (!square && k % 2 == 1) {
            continue;  // rectangular boards keep the shape-preserving transforms only
        }
        std::vector<int> perm(geo.num_edges);
        for (int e = 0; e < geo.num_edges; ++e) {
            auto [r, c] = transform_cell(geo.edge_rc[e].first, geo.edge_rc[e].second, height,
                                         width, k);
            perm[e] = geo.lattice_edge[r * width + c];
        }
        perms_.push_back(perm);
    }
    log2_size_ = 1;
    while ((std::size_t{1} << log2_size_) < table_entries) {
        ++log2_size_;
    }
    table_.assign(std::size_t{1} << log2_size_, Entry{0, 0, kEmpty});
}

uint64_t BoxesSolver::canonical(uint64_t mask) const {
    uint64_t best = ~uint64_t{0};
    for (const auto& perm : perms_) {
        uint64_t image = 0;
        for (uint64_t bits = mask; bits; bits &= bits - 1) {
            image |= chains::bit(perm[__builtin_ctzll(bits)]);
        }
        best = std::min(best, image);
    }
    return best;
}

int BoxesSolver::loony_value(std::vector<int> chains, std::vector<int> loops) {
    std::sort(chains.begin(), chains.end());
    std::sort(loops.begin(), loops.end());
    auto key = std::make_pair(chains, loops);
    auto found = loony_cache_.find(key);
    if (found != loony_cache_.end()) {
        return found->second;
    }
    int best = INT_MIN;
    for (std::size_t i = 0; i < chains.size(); ++i) {
        if (i > 0 && chains[i - 1] == chains[i]) {
            continue;
        }
        std::vector<int> rest_chains = chains;
        rest_chains.erase(rest_chains.begin() + static_cast<std::ptrdiff_t>(i));
        const int size = chains[i];
        const int rest = loony_value(rest_chains, loops);
        const int take_all = -size - rest;
        best = std::max(best, size <= 2 ? take_all : std::min(take_all, 4 - size + rest));
    }
    for (std::size_t i = 0; i < loops.size(); ++i) {
        if (i > 0 && loops[i - 1] == loops[i]) {
            continue;
        }
        std::vector<int> rest_loops = loops;
        rest_loops.erase(rest_loops.begin() + static_cast<std::ptrdiff_t>(i));
        const int size = loops[i];
        const int rest = loony_value(chains, rest_loops);
        best = std::max(best, std::min(-size - rest, 8 - size + rest));
    }
    const int value = best == INT_MIN ? 0 : best;
    loony_cache_[key] = value;
    return value;
}

std::vector<int> BoxesSolver::moves(uint64_t quiet, const std::vector<int>& deg,
                                    const std::vector<chains::Component>& comps) const {
    const BoxesGeometry& geo = proto_.geometry();
    std::vector<bool> drop(geo.num_edges, false);
    for (const auto& comp : comps) {
        if (!comp.independent()) {
            continue;
        }
        std::vector<int> edges;
        for (int b : comp.boxes) {
            for (int e : chains::undrawn_sides(quiet, geo, b)) {
                edges.push_back(e);
            }
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        int keep = edges[0];
        if (!comp.is_loop && comp.size() == 2) {  // hard-hearted handout: the middle edge
            keep = chains::shared_edge(geo, comp.boxes[0], comp.boxes[1]);
        }
        for (int e : edges) {
            drop[e] = drop[e] || e != keep;
        }
    }
    std::vector<int> safe;
    std::vector<int> loony;
    for (int e = 0; e < geo.num_edges; ++e) {
        if (chains::drawn(quiet, e) || drop[e]) {
            continue;
        }
        bool is_safe = true;
        for (int b : geo.edge_boxes[e]) {
            is_safe = is_safe && !(b >= 0 && deg[b] == 2);
        }
        (is_safe ? safe : loony).push_back(e);
    }
    safe.insert(safe.end(), loony.begin(), loony.end());
    return safe;
}

int BoxesSolver::child_after_take(uint64_t quiet, const chains::Decision& decision, int alpha,
                                  int beta) {
    uint64_t after = quiet;
    for (int e : decision.take) {
        after |= chains::bit(e);
    }
    const int gained = chains::completed(quiet, after, proto_.geometry());
    return gained + search(after, alpha - gained, beta - gained);  // same mover continues
}

int BoxesSolver::search(uint64_t mask, int alpha, int beta) {
    if (aborted_) {
        return 0;
    }
    if (++nodes_ > budget_) {
        aborted_ = true;
        return 0;
    }
    const BoxesGeometry& geo = proto_.geometry();
    const uint64_t key = canonical(mask);
    Entry& slot = table_[index(key)];
    if (slot.flag != kEmpty && slot.key == key) {
        if (slot.flag == kExact) {
            return slot.value;
        }
        if (slot.flag == kLower) {
            alpha = std::max(alpha, static_cast<int>(slot.value));
        } else {
            beta = std::min(beta, static_cast<int>(slot.value));
        }
        if (alpha >= beta) {
            return slot.value;
        }
    }
    const int alpha0 = alpha;
    const int beta0 = beta;

    std::vector<int> prefix;
    chains::Decision decision;
    const uint64_t quiet = chains::collapse_mask(mask, geo, prefix, decision);
    const int gained = chains::completed(mask, quiet, geo);
    int best;
    if (quiet == full_) {
        best = gained;
    } else if (decision.control >= 0) {
        best = gained + child_after_take(quiet, decision, alpha - gained, beta - gained);
        if (aborted_) {
            return 0;
        }
        alpha = std::max(alpha, best);
        if (alpha < beta) {
            const int control = gained - search(quiet | chains::bit(decision.control),
                                                -(beta - gained), -(alpha - gained));
            if (aborted_) {
                return 0;
            }
            best = std::max(best, control);
        }
    } else {
        const std::vector<int> deg = chains::degrees(quiet, geo);
        const std::vector<chains::Component> comps = chains::components(quiet, geo, deg);
        bool simple = *std::max_element(deg.begin(), deg.end()) <= 2;
        for (const auto& c : comps) {
            simple = simple && c.independent();
        }
        if (simple) {
            std::vector<int> chain_sizes;
            std::vector<int> loop_sizes;
            for (const auto& c : comps) {
                (c.is_loop ? loop_sizes : chain_sizes).push_back(c.size());
            }
            best = gained + loony_value(chain_sizes, loop_sizes);
        } else {
            best = -kInf;
            for (int e : moves(quiet, deg, comps)) {
                const int v = gained - search(quiet | chains::bit(e), -(beta - gained),
                                              -(alpha - gained));
                if (aborted_) {
                    return 0;
                }
                best = std::max(best, v);
                alpha = std::max(alpha, v);
                if (alpha >= beta) {
                    break;
                }
            }
        }
    }
    slot.key = key;
    slot.value = static_cast<int16_t>(best);
    slot.flag = best <= alpha0 ? kUpper : (best >= beta0 ? kLower : kExact);
    return best;
}

int BoxesSolver::value(uint64_t mask) {
    nodes_ = 0;
    budget_ = ~uint64_t{0};
    aborted_ = false;
    return search(mask, -kInf, kInf);
}

std::optional<int> BoxesSolver::value_within(uint64_t mask, uint64_t max_nodes) {
    nodes_ = 0;
    budget_ = max_nodes;
    aborted_ = false;
    const int v = search(mask, -kInf, kInf);
    if (aborted_) {
        return std::nullopt;
    }
    return v;
}

int BoxesSolver::best_edge(const BoxesBoard& board) {
    const BoxesGeometry& geo = proto_.geometry();
    std::vector<int> prefix;
    chains::Decision decision;
    const uint64_t quiet = chains::collapse_mask(board.edges(), geo, prefix, decision);
    if (!prefix.empty()) {
        return prefix[0];
    }
    if (decision.control >= 0) {
        uint64_t after = quiet;
        for (int e : decision.take) {
            after |= chains::bit(e);
        }
        const int take = chains::completed(quiet, after, geo) + value(after);
        const int control = -value(quiet | chains::bit(decision.control));
        return take >= control ? decision.take[0] : decision.control;
    }
    int best_edge = -1;
    int best = INT_MIN;
    for (int e = 0; e < geo.num_edges; ++e) {
        if (chains::drawn(quiet, e)) {
            continue;
        }
        const int v = -value(quiet | chains::bit(e));
        if (v > best) {
            best = v;
            best_edge = e;
        }
    }
    return best_edge;
}

}  // namespace alpha_go
