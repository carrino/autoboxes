#include "boxes_classic.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace alpha_go {

BoxesClassic::BoxesClassic(int rows, int cols, double seconds, int solver_max_undrawn,
                           std::size_t table_entries, std::size_t solver_entries)
    : solver_(rows, cols, solver_entries), full_(chains::full_mask(solver_.geometry())),
      seconds_(seconds), solver_max_undrawn_(solver_max_undrawn) {
    if (!std::isfinite(seconds) || seconds < 0 || solver_max_undrawn < 0 || table_entries < 2) {
        throw std::invalid_argument("BoxesClassic needs nonnegative time/reach and >=2 TT entries");
    }
    while ((std::size_t{1} << log2_size_) < table_entries) {
        ++log2_size_;
    }
    table_.assign(std::size_t{1} << log2_size_, Entry{0, 0, 0, kEmpty, -1});
    history_.assign(solver_.geometry().num_edges, 0);
    rank_.resize(solver_.geometry().num_edges);
    std::iota(rank_.begin(), rank_.end(), 0);
}

int BoxesClassic::quiet_estimate(uint64_t mask, const std::vector<int>& deg,
                                 const std::vector<chains::Component>& comps) {
    std::vector<int> chain_sizes;
    std::vector<int> loops;
    bool simple = *std::max_element(deg.begin(), deg.end()) <= 2;
    for (const auto& c : comps) {
        (c.is_loop ? loops : chain_sizes).push_back(c.size());
        simple = simple && c.independent();
    }
    if (simple) {
        return kScale * solver_.loony_value(chain_sizes, loops);
    }
    int nchain = 0, length = 0, longest = 0;
    for (int size : chain_sizes) {
        if (size >= 3) {
            ++nchain;
            length += size;
            longest = std::max(longest, size);
        }
    }
    const int loop_length = std::accumulate(loops.begin(), loops.end(), 0);
    const int nloop = static_cast<int>(loops.size());
    const int bonus = nchain == 0 ? (nloop == 0 ? 0 : 8)
                                  : (nloop > 0 && longest == 3 ? 6 : 4);
    const int controlled = length - 4 * nchain + loop_length - 8 * nloop + bonus;
    const int remaining = static_cast<int>(std::count_if(deg.begin(), deg.end(),
                                                        [](int d) { return d > 0; }));
    const int junctions = static_cast<int>(std::count_if(deg.begin(), deg.end(),
                                                        [](int d) { return d >= 3; }));
    const int potential = std::max(0, controlled + remaining - length - loop_length);
    const int advantage = std::min(kScale * remaining, kScale * potential / (1 + junctions));
    const int safe_turns = std::max(0, __builtin_popcountll(full_ & ~mask) - remaining - nchain);
    return safe_turns % 2 ? advantage : -advantage;
}

int BoxesClassic::estimate(uint64_t mask) {
    const auto& geo = solver_.geometry();
    std::vector<int> prefix;
    chains::Decision decision;
    const uint64_t quiet = chains::collapse_mask(mask, geo, prefix, decision);
    const int gained = kScale * chains::completed(mask, quiet, geo);
    if (quiet == full_) {
        return gained;
    }
    if (decision.control >= 0) {
        uint64_t after = quiet;
        for (int e : decision.take) {
            after |= chains::bit(e);
        }
        const int take = kScale * chains::completed(quiet, after, geo) + estimate(after);
        return gained + std::max(take, -estimate(quiet | chains::bit(decision.control)));
    }
    const auto deg = chains::degrees(quiet, geo);
    return gained + quiet_estimate(quiet, deg, chains::components(quiet, geo, deg));
}

double BoxesClassic::evaluate(uint64_t mask) {
    return estimate(mask) / static_cast<double>(kScale);
}

bool BoxesClassic::expired() {
    aborted_ = aborted_ || (timed_ && std::chrono::steady_clock::now() >= deadline_);
    return aborted_;
}

std::vector<int> BoxesClassic::moves(uint64_t mask, const std::vector<int>& deg,
                                     const std::vector<chains::Component>& comps,
                                     int first) const {
    const auto& geo = solver_.geometry();
    const auto drop = chains::equivalent_drop(mask, geo, comps);
    std::vector<int> legal;
    std::vector<int> danger(geo.num_edges, 0);
    for (int e = 0; e < geo.num_edges; ++e) {
        if (!chains::drawn(mask, e) && !drop[e]) {
            legal.push_back(e);
            for (int box : geo.edge_boxes[e]) {
                danger[e] += box >= 0 && deg[box] == 2 ? 1 : 0;
            }
        }
    }
    std::sort(legal.begin(), legal.end(), [&](int a, int b) {
        if (a == first || b == first) {
            return a == first && b != first;
        }
        if (danger[a] != danger[b]) {
            return danger[a] < danger[b];
        }
        return history_[a] != history_[b] ? history_[a] > history_[b] : rank_[a] < rank_[b];
    });
    return legal;
}

int BoxesClassic::search(uint64_t mask, int depth, int alpha, int beta, int& move, bool root) {
    move = -1;
    ++nodes_;
    if (aborted_ || ((nodes_ & 63) == 1 && expired())) {
        return 0;
    }
    if (mask == full_) {
        return 0;
    }
    Entry& slot = table_[index(mask)];
    int first = -1;
    if (slot.flag != kEmpty && slot.key == mask) {
        first = slot.move;
        // A solved child entry has no selected move, so the root still searches its actions.
        if ((!root || first >= 0) && (slot.depth == depth || slot.depth == kSolved)) {
            move = first;
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
    }
    if (!root && solver_max_undrawn_ > 0 &&
        __builtin_popcountll(full_ & ~mask) <= solver_max_undrawn_) {
        if (expired()) {
            return 0;
        }
        const auto exact = solver_.value_within(mask, 1024);
        solver_nodes_ += solver_.nodes();
        if (expired()) {
            return 0;
        }
        if (exact) {
            slot = Entry{mask, static_cast<int16_t>(*exact * kScale), kSolved, kExact, -1};
            return *exact * kScale;
        }
    }
    if (depth == 0) {
        return estimate(mask);
    }
    const int alpha0 = alpha, beta0 = beta;
    const auto& geo = solver_.geometry();
    std::vector<int> prefix;
    chains::Decision decision;
    const uint64_t quiet = chains::collapse_mask(mask, geo, prefix, decision);
    const int gained = kScale * chains::completed(mask, quiet, geo);
    int best = -kInf;
    int ignored = -1;
    if (quiet == full_) {
        best = gained;
    } else if (decision.control >= 0) {
        uint64_t after = quiet;
        for (int e : decision.take) {
            after |= chains::bit(e);
        }
        const int take_gain = gained + kScale * chains::completed(quiet, after, geo);
        best = take_gain + search(after, depth, alpha - take_gain, beta - take_gain, ignored);
        move = decision.take[0];
        alpha = std::max(alpha, best);
        if (!aborted_ && alpha < beta) {
            const int control = gained - search(quiet | chains::bit(decision.control), depth - 1,
                                                gained - beta, gained - alpha, ignored);
            if (control > best) {
                best = control;
                move = decision.control;
            }
        }
    } else {
        const auto deg = chains::degrees(quiet, geo);
        const auto comps = chains::components(quiet, geo, deg);
        for (int e : moves(quiet, deg, comps, first)) {
            const int v = gained - search(quiet | chains::bit(e), depth - 1,
                                          gained - beta, gained - alpha, ignored);
            if (aborted_) {
                return 0;
            }
            if (v > best) {
                best = v;
                move = e;
            }
            alpha = std::max(alpha, v);
            if (alpha >= beta) {
                history_[e] += static_cast<uint32_t>(depth * depth);
                if (history_[e] > (uint32_t{1} << 28)) {
                    for (auto& h : history_) {
                        h >>= 1;
                    }
                }
                break;
            }
        }
    }
    if (aborted_) {
        return 0;
    }
    if (!prefix.empty()) {
        move = prefix[0];
    }
    const auto flag = best <= alpha0 ? kUpper : (best >= beta0 ? kLower : kExact);
    slot = Entry{mask, static_cast<int16_t>(best), static_cast<int8_t>(depth), flag,
                 static_cast<int8_t>(move)};
    return best;
}

double BoxesClassic::value(uint64_t mask, int depth) {
    if (depth < 0 || depth > 64) {
        throw std::invalid_argument("depth must be in 0..64");
    }
    timed_ = false;
    aborted_ = false;
    nodes_ = solver_nodes_ = 0;
    int move = -1;
    return search(mask, depth, -kInf, kInf, move, true) / static_cast<double>(kScale);
}

int BoxesClassic::best_edge(uint64_t mask, uint64_t seed) {
    const auto start = std::chrono::steady_clock::now();
    deadline_ = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(seconds_));
    timed_ = true;
    aborted_ = false;
    nodes_ = solver_nodes_ = 0;
    completed_depth_ = 0;
    last_value_ = estimate(mask);
    if (mask == full_) {
        return -1;
    }
    std::vector<int> prefix;
    chains::Decision decision;
    chains::collapse_mask(mask, solver_.geometry(), prefix, decision);
    if (!prefix.empty()) {
        return prefix[0];
    }
    std::iota(rank_.begin(), rank_.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(rank_.begin(), rank_.end(), rng);
    const auto deg = chains::degrees(mask, solver_.geometry());
    const auto comps = chains::components(mask, solver_.geometry(), deg);
    int best = decision.control >= 0 ? decision.take[0] : moves(mask, deg, comps, -1)[0];
    const int max_depth = __builtin_popcountll(full_ & ~mask);
    for (int depth = 1; depth <= max_depth && !expired(); ++depth) {
        int move = -1;
        const int v = search(mask, depth, -kInf, kInf, move, true);
        if (aborted_) {
            break;
        }
        if (move >= 0) {
            best = move;
            last_value_ = v;
            completed_depth_ = depth;
        }
    }
    return best;
}

}  // namespace alpha_go
