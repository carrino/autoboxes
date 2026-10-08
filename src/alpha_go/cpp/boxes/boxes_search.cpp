#include "boxes_search.h"

#include <algorithm>

namespace alpha_go {

namespace {

bool drawn(uint64_t mask, int edge) { return (mask >> edge) & 1; }

std::vector<int> undrawn_sides(uint64_t mask, const BoxesGeometry& geo, int box) {
    std::vector<int> out;
    for (int e : geo.box_edges[box]) {
        if (!drawn(mask, e)) {
            out.push_back(e);
        }
    }
    return out;
}

int shared_edge(const BoxesGeometry& geo, int a, int b) {
    for (int e : geo.box_edges[a]) {
        if (geo.edge_boxes[e][0] == b || geo.edge_boxes[e][1] == b) {
            return e;
        }
    }
    return -1;
}

}  // namespace

BoxesSearchState::BoxesSearchState(const BoxesBoard& board) : board_(board) {
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

std::vector<BoxesSearchState::Component> BoxesSearchState::components(
    uint64_t mask, const std::vector<int>& deg) const {
    const BoxesGeometry& geo = board_.geometry();
    std::vector<bool> seen(geo.num_boxes, false);
    std::vector<Component> out;

    // Follow degree-2 boxes from `start` across `via`; returns the end marker and
    // appends the boxes walked to `path`.
    auto walk = [&](int start, int via, std::vector<int>& path) -> int {
        int box = start;
        int edge = via;
        while (true) {
            const auto& pair = geo.edge_boxes[edge];
            int nxt = (pair[0] == box) ? pair[1] : pair[0];
            if (nxt < 0) {
                return GROUND;
            }
            if (deg[nxt] >= 3) {
                return nxt;
            }
            if (seen[nxt]) {
                return deg[nxt] == 1 ? OPEN : nxt;
            }
            seen[nxt] = true;
            path.push_back(nxt);
            std::vector<int> others;
            for (int e : undrawn_sides(mask, geo, nxt)) {
                if (e != edge) {
                    others.push_back(e);
                }
            }
            if (others.empty()) {
                return OPEN;
            }
            box = nxt;
            edge = others[0];
        }
    };

    for (int b = 0; b < geo.num_boxes; ++b) {
        if (seen[b] || (deg[b] != 1 && deg[b] != 2)) {
            continue;
        }
        seen[b] = true;
        std::vector<int> edges = undrawn_sides(mask, geo, b);
        std::vector<int> left;
        int left_end = OPEN;
        if (deg[b] == 2) {
            left_end = walk(b, edges[0], left);
            if (left_end == b) {
                Component loop{{b}, true, b, b};
                loop.boxes.insert(loop.boxes.end(), left.begin(), left.end());
                out.push_back(loop);
                continue;
            }
        }
        std::vector<int> right;
        int right_end = walk(b, edges.back(), right);
        Component comp{{}, false, left_end, right_end};
        comp.boxes.assign(left.rbegin(), left.rend());
        comp.boxes.push_back(b);
        comp.boxes.insert(comp.boxes.end(), right.begin(), right.end());
        out.push_back(comp);
    }
    return out;
}

std::vector<int> BoxesSearchState::open_end_first(
    const Component& comp, const std::vector<int>& deg) const {
    if (deg[comp.boxes.front()] == 1) {
        return comp.boxes;
    }
    return std::vector<int>(comp.boxes.rbegin(), comp.boxes.rend());
}

std::vector<int> BoxesSearchState::take_sequence(uint64_t mask, const std::vector<int>& boxes) const {
    std::vector<int> edges;
    for (int box : boxes) {
        std::vector<int> undrawn = undrawn_sides(mask, board_.geometry(), box);
        if (!undrawn.empty()) {  // otherwise the previous edge completed this box as well
            edges.push_back(undrawn[0]);
            mask |= uint64_t{1} << undrawn[0];
        }
    }
    return edges;
}

int BoxesSearchState::control_edge(
    uint64_t mask, const Component& comp, const std::vector<int>& deg) const {
    const BoxesGeometry& geo = board_.geometry();
    std::vector<int> boxes = open_end_first(comp, deg);
    bool both_open = comp.end0 == OPEN && comp.end1 == OPEN;
    if (both_open && comp.boxes.size() == 4) {
        return shared_edge(geo, boxes[1], boxes[2]);
    }
    if (!both_open && comp.boxes.size() == 2) {
        int link = shared_edge(geo, boxes[0], boxes[1]);
        for (int e : undrawn_sides(mask, geo, boxes[1])) {
            if (e != link) {
                return e;
            }
        }
    }
    return -1;
}

void BoxesSearchState::collapse() {
    const BoxesGeometry& geo = board_.geometry();
    while (!board_.is_game_over()) {
        uint64_t mask = board_.edges();
        std::vector<int> deg(geo.num_boxes);
        for (int b = 0; b < geo.num_boxes; ++b) {
            deg[b] = 4 - board_.sides(b);
        }
        std::vector<Component> opened;
        for (const Component& c : components(mask, deg)) {
            if (c.opened()) {
                opened.push_back(c);
            }
        }
        if (opened.empty()) {
            return;
        }
        auto remainder = [](const Component& c) { return (c.end0 == OPEN && c.end1 == OPEN) ? 4 : 2; };
        // Keep for last the component that offers control at its remainder (chains
        // before loops: the cheaper sacrifice); everything else is taken in full.
        const Component* last = nullptr;
        for (const Component& c : opened) {
            bool candidate = static_cast<int>(c.boxes.size()) >= remainder(c);
            if (candidate && (last == nullptr || remainder(c) < remainder(*last))) {
                last = &c;
            }
        }
        bool took_other = false;
        for (const Component& c : opened) {
            if (&c == last) {
                continue;
            }
            for (int e : take_sequence(mask, open_end_first(c, deg))) {
                board_.play_edge(e);
                prefix_.push_back(e);
            }
            took_other = true;
            break;  // positions changed: recompute components
        }
        if (took_other) {
            continue;
        }
        std::vector<int> boxes = open_end_first(*last, deg);
        int surplus = static_cast<int>(boxes.size()) - remainder(*last);
        if (surplus > 0) {
            std::vector<int> head(boxes.begin(), boxes.begin() + surplus);
            for (int e : take_sequence(mask, head)) {
                board_.play_edge(e);
                prefix_.push_back(e);
            }
            continue;
        }
        int control = control_edge(mask, *last, deg);
        if (control < 0) {
            for (int e : take_sequence(mask, boxes)) {
                board_.play_edge(e);
                prefix_.push_back(e);
            }
            continue;
        }
        take_ = take_sequence(mask, boxes);
        control_ = control;
        has_decision_ = true;
        return;
    }
}

}  // namespace alpha_go
