#include "boxes_alphabeta.h"

#include <algorithm>
#include <climits>

#include "boxes/boxes_chains.h"

namespace alpha_go {

namespace {
constexpr int kInf = 1000000;
}

BoxesAlphaBeta::BoxesAlphaBeta(int rows, int cols, std::size_t table_entries)
    : proto_(rows, cols), full_(chains::full_mask(proto_.geometry())) {
    log2_size_ = 1;
    while ((std::size_t{1} << log2_size_) < table_entries) {
        ++log2_size_;
    }
    table_.assign(std::size_t{1} << log2_size_, Entry{0, 0, 0, kEmpty});
}

int BoxesAlphaBeta::sides(uint64_t mask, int box) const {
    int drawn = 0;
    for (int e : proto_.geometry().box_edges[box]) {
        drawn += chains::drawn(mask, e) ? 1 : 0;
    }
    return drawn;
}

int BoxesAlphaBeta::classify(uint64_t mask, int edge) const {
    int most = 0;
    for (int b : proto_.geometry().edge_boxes[edge]) {
        if (b >= 0) {
            most = std::max(most, sides(mask, b));
        }
    }
    most += 1;
    return most == 4 ? 0 : (most == 3 ? 2 : 1);
}

int BoxesAlphaBeta::captures(uint64_t mask, int edge) const {
    int count = 0;
    for (int b : proto_.geometry().edge_boxes[edge]) {
        count += (b >= 0 && sides(mask, b) == 3) ? 1 : 0;
    }
    return count;
}

int BoxesAlphaBeta::greedy_haul(uint64_t mask) const {
    // Take the capture completing the most boxes until nothing is capturable.
    const BoxesGeometry& geo = proto_.geometry();
    int haul = 0;
    while (mask != full_) {
        int best_edge = -1;
        int best = 0;
        for (int e = 0; e < geo.num_edges; ++e) {
            if (!chains::drawn(mask, e)) {
                const int c = captures(mask, e);
                if (c > best) {
                    best = c;
                    best_edge = e;
                }
            }
        }
        if (best_edge < 0) {
            break;
        }
        haul += best;
        mask |= chains::bit(best_edge);
    }
    return haul;
}

std::vector<int> BoxesAlphaBeta::ordered_moves(uint64_t mask, std::mt19937_64& rng) const {
    std::vector<int> legal;
    for (int e = 0; e < proto_.geometry().num_edges; ++e) {
        if (!chains::drawn(mask, e)) {
            legal.push_back(e);
        }
    }
    std::shuffle(legal.begin(), legal.end(), rng);
    std::stable_sort(legal.begin(), legal.end(),
                     [&](int a, int b) { return classify(mask, a) < classify(mask, b); });
    return legal;
}

int BoxesAlphaBeta::search(uint64_t mask, int depth, int alpha, int beta, std::mt19937_64& rng) {
    ++nodes_;
    if (mask == full_) {
        return 0;
    }
    if (depth == 0) {
        return greedy_haul(mask);
    }
    Entry& slot = table_[index(mask, depth)];
    if (slot.flag != kEmpty && slot.key == mask && slot.depth == depth) {
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
    int best = -kInf;
    for (int e : ordered_moves(mask, rng)) {
        const int gained = captures(mask, e);
        const uint64_t child = mask | chains::bit(e);
        int value;
        if (gained > 0) {  // the mover keeps the turn; captures do not consume depth
            value = gained + search(child, depth, alpha - gained, beta - gained, rng);
        } else {
            value = -search(child, depth - 1, -beta, -alpha, rng);
        }
        best = std::max(best, value);
        alpha = std::max(alpha, value);
        if (alpha >= beta) {
            break;
        }
    }
    slot.key = mask;
    slot.depth = static_cast<int8_t>(depth);
    slot.value = static_cast<int16_t>(best);
    slot.flag = best <= alpha0 ? kUpper : (best >= beta ? kLower : kExact);
    return best;
}

int BoxesAlphaBeta::value(uint64_t mask, int depth) {
    nodes_ = 0;
    std::mt19937_64 rng(0);
    return search(mask, depth, -kInf, kInf, rng);
}

int BoxesAlphaBeta::best_edge(uint64_t mask, int depth, uint64_t seed) {
    nodes_ = 0;
    std::mt19937_64 rng(seed);
    int best_edge = -1;
    int best = INT_MIN;
    for (int e : ordered_moves(mask, rng)) {
        const int gained = captures(mask, e);
        const uint64_t child = mask | chains::bit(e);
        const int value = gained > 0 ? gained + search(child, depth, -kInf, kInf, rng)
                                     : -search(child, depth - 1, -kInf, kInf, rng);
        if (value > best) {
            best = value;
            best_edge = e;
        }
    }
    return best_edge;
}

}  // namespace alpha_go
