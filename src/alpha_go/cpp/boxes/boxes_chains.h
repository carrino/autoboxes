#pragma once

#include <cstdint>
#include <vector>

#include "boxes/boxes_game.h"

namespace alpha_go {
namespace chains {

// Chain and loop analysis on the strings-and-coins dual, mirroring
// src/alpha_go/boxes/chains.py and the forced-move collapse of forced.py, as pure
// functions of an edge mask and the geometry. Shared by the search state and the
// endgame solver.

constexpr int GROUND = -1;
constexpr int OPEN = -2;

struct Component {
    std::vector<int> boxes;  // path order (any order for a loop)
    bool is_loop;
    int end0;  // GROUND, a junction box index, or OPEN (a degree-1 end)
    int end1;
    bool opened() const { return end0 == OPEN || end1 == OPEN; }
    bool independent() const { return is_loop || (end0 == GROUND && end1 == GROUND); }
    int size() const { return static_cast<int>(boxes.size()); }
};

// Pending take-all / keep-control choice on the last opened component.
struct Decision {
    std::vector<int> take;  // edges that capture the remainder
    int control = -1;       // the double-dealing edge, or -1 when there is no decision
};

inline bool drawn(uint64_t mask, int edge) { return (mask >> edge) & 1; }
inline uint64_t bit(int edge) { return uint64_t{1} << edge; }
inline uint64_t full_mask(const BoxesGeometry& geo) {
    return geo.num_edges == 64 ? ~uint64_t{0} : (uint64_t{1} << geo.num_edges) - 1;
}

std::vector<int> undrawn_sides(uint64_t mask, const BoxesGeometry& geo, int box);
int shared_edge(const BoxesGeometry& geo, int a, int b);
std::vector<int> degrees(uint64_t mask, const BoxesGeometry& geo);
std::vector<Component> components(uint64_t mask, const BoxesGeometry& geo,
                                  const std::vector<int>& deg);
std::vector<int> open_end_first(const Component& comp, const std::vector<int>& deg);
std::vector<int> take_sequence(uint64_t mask, const BoxesGeometry& geo,
                               const std::vector<int>& boxes);
int control_edge(uint64_t mask, const BoxesGeometry& geo, const Component& comp,
                 const std::vector<int>& deg);
// Boxes completed between two masks.
int completed(uint64_t before, uint64_t after, const BoxesGeometry& geo);
// Forced-move collapse: auto-capture for the side to move until nothing is forced.
// Returns the resulting mask, appends the forced edges to `prefix` in order, and fills
// `decision` (control >= 0) when the last remainder offers the keep-control choice.
uint64_t collapse_mask(uint64_t mask, const BoxesGeometry& geo, std::vector<int>& prefix,
                       Decision& decision);

}  // namespace chains
}  // namespace alpha_go
