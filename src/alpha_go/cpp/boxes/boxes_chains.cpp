#include "boxes_chains.h"

#include <algorithm>

namespace alpha_go {
namespace chains {

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

std::vector<int> degrees(uint64_t mask, const BoxesGeometry& geo) {
    std::vector<int> deg(geo.num_boxes, 0);
    for (int b = 0; b < geo.num_boxes; ++b) {
        for (int e : geo.box_edges[b]) {
            deg[b] += drawn(mask, e) ? 0 : 1;
        }
    }
    return deg;
}

std::vector<Component> components(uint64_t mask, const BoxesGeometry& geo,
                                  const std::vector<int>& deg) {
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

std::vector<int> open_end_first(const Component& comp, const std::vector<int>& deg) {
    if (deg[comp.boxes.front()] == 1) {
        return comp.boxes;
    }
    return std::vector<int>(comp.boxes.rbegin(), comp.boxes.rend());
}

std::vector<int> take_sequence(uint64_t mask, const BoxesGeometry& geo,
                               const std::vector<int>& boxes) {
    std::vector<int> edges;
    for (int box : boxes) {
        std::vector<int> undrawn = undrawn_sides(mask, geo, box);
        if (!undrawn.empty()) {  // otherwise the previous edge completed this box as well
            edges.push_back(undrawn[0]);
            mask |= bit(undrawn[0]);
        }
    }
    return edges;
}

int control_edge(uint64_t mask, const BoxesGeometry& geo, const Component& comp,
                 const std::vector<int>& deg) {
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

std::vector<bool> equivalent_drop(uint64_t mask, const BoxesGeometry& geo,
                                  const std::vector<Component>& comps) {
    std::vector<bool> drop(geo.num_edges, false);
    for (const auto& comp : comps) {
        if (!comp.independent()) {
            continue;
        }
        std::vector<int> edges;
        for (int b : comp.boxes) {
            for (int e : undrawn_sides(mask, geo, b)) {
                edges.push_back(e);
            }
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        int keep = edges[0];
        if (!comp.is_loop && comp.boxes.size() == 2) {  // hard-hearted handout: the middle edge
            keep = shared_edge(geo, comp.boxes[0], comp.boxes[1]);
        }
        for (int e : edges) {
            drop[e] = drop[e] || e != keep;
        }
    }
    return drop;
}

int completed(uint64_t before, uint64_t after, const BoxesGeometry& geo) {
    int count = 0;
    for (const auto& sides : geo.box_edges) {
        bool was = true;
        bool now = true;
        for (int e : sides) {
            was = was && drawn(before, e);
            now = now && drawn(after, e);
        }
        count += (now && !was) ? 1 : 0;
    }
    return count;
}

uint64_t collapse_mask(uint64_t mask, const BoxesGeometry& geo, std::vector<int>& prefix,
                       Decision& decision) {
    const uint64_t full = full_mask(geo);
    decision.take.clear();
    decision.control = -1;
    auto remainder = [](const Component& c) {
        return (c.end0 == OPEN && c.end1 == OPEN) ? 4 : 2;
    };
    while (mask != full) {
        std::vector<int> deg = degrees(mask, geo);
        std::vector<Component> opened;
        for (const Component& c : components(mask, geo, deg)) {
            if (c.opened()) {
                opened.push_back(c);
            }
        }
        if (opened.empty()) {
            return mask;
        }
        // Keep for last the component that offers control at its remainder (chains
        // before loops: the cheaper sacrifice); everything else is taken in full.
        const Component* last = nullptr;
        for (const Component& c : opened) {
            bool candidate = c.size() >= remainder(c);
            if (candidate && (last == nullptr || remainder(c) < remainder(*last))) {
                last = &c;
            }
        }
        std::vector<int> taken;
        bool took_other = false;
        for (const Component& c : opened) {
            if (&c == last) {
                continue;
            }
            taken = take_sequence(mask, geo, open_end_first(c, deg));
            took_other = true;
            break;  // positions changed: recompute components
        }
        if (!took_other) {
            std::vector<int> boxes = open_end_first(*last, deg);
            int surplus = last->size() - remainder(*last);
            if (surplus > 0) {
                taken = take_sequence(mask, geo,
                                      std::vector<int>(boxes.begin(), boxes.begin() + surplus));
            } else {
                int control = control_edge(mask, geo, *last, deg);
                if (control >= 0) {
                    decision.take = take_sequence(mask, geo, boxes);
                    decision.control = control;
                    return mask;
                }
                taken = take_sequence(mask, geo, boxes);
            }
        }
        for (int e : taken) {
            mask |= bit(e);
        }
        prefix.insert(prefix.end(), taken.begin(), taken.end());
    }
    return mask;
}

}  // namespace chains
}  // namespace alpha_go
