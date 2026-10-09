#include "boxes_game.h"

#include <catch2/catch_test_macros.hpp>

using namespace alpha_go;

TEST_CASE("BoxesBoard construction", "[boxes_game]") {
    BoxesBoard board(3);
    REQUIRE(board.rows() == 3);
    REQUIRE(board.cols() == 3);
    REQUIRE(board.num_edges() == 24);
    REQUIRE(board.num_boxes() == 9);
    REQUIRE(board.to_play() == BoxesBoard::PLAYER_1);
    REQUIRE(board.get_legal_moves_flat().size() == 24);
    REQUIRE_FALSE(board.is_game_over());

    BoxesBoard rect(2, 3);
    REQUIRE(rect.num_edges() == 17);
    REQUIRE_THROWS(BoxesBoard(6, 6));  // 84 edges exceeds the 64-bit mask
}

TEST_CASE("Edge geometry", "[boxes_game]") {
    BoxesBoard board(2, 3);
    // h(1,1) has index 1*3 + 1 = 4 and sits at lattice (2, 3); v(0,2) is (2+1)*3 + 2 = 11 at (1, 4).
    REQUIRE(board.row_col(4) == std::make_pair(2, 3));
    REQUIRE(board.row_col(11) == std::make_pair(1, 4));
    REQUIRE(board.edge_index(2, 3) == 4);
    REQUIRE(board.edge_index(0, 0) == -1);  // dot
    REQUIRE(board.edge_index(1, 1) == -1);  // box
    REQUIRE(board.edge_index(-1, 0) == -1);
    REQUIRE(board.edge_index(0, 99) == -1);
}

TEST_CASE("Non-capturing move switches player", "[boxes_game]") {
    BoxesBoard board(2);
    REQUIRE(board.play_edge(0));
    REQUIRE(board.to_play() == BoxesBoard::PLAYER_2);
    REQUIRE_FALSE(board.is_legal_edge(0));
    REQUIRE_FALSE(board.play_edge(0));
    REQUIRE(board.move_count() == 1);
}

TEST_CASE("Capture keeps the turn", "[boxes_game]") {
    BoxesBoard board(2, 2);
    const auto& sides = board.geometry().box_edges[0];
    board.play_edge(sides[0]);  // P1
    board.play_edge(sides[1]);  // P2
    board.play_edge(sides[2]);  // P1
    board.play_edge(board.geometry().box_edges[3][0]);  // P2 elsewhere
    REQUIRE(board.to_play() == BoxesBoard::PLAYER_1);
    board.play_edge(sides[3]);  // P1 completes box 0
    REQUIRE(board.owner(0) == BoxesBoard::PLAYER_1);
    REQUIRE(board.boxes(0) == 1);
    REQUIRE(board.to_play() == BoxesBoard::PLAYER_1);
    REQUIRE(board.margin() == 1);
    board.play_edge(board.geometry().box_edges[3][1]);  // no capture
    REQUIRE(board.to_play() == BoxesBoard::PLAYER_2);
    REQUIRE(board.margin() == -1);
}

TEST_CASE("Double capture scores two and ends the game", "[boxes_game]") {
    BoxesBoard board(1, 2);
    for (int e : {0, 1, 2, 3, 4, 6}) {
        board.play_edge(e);
    }
    REQUIRE(board.to_play() == BoxesBoard::PLAYER_1);
    board.play_edge(5);
    REQUIRE(board.boxes(0) == 2);
    REQUIRE(board.is_game_over());
    REQUIRE(board.get_winner() == BoxesBoard::PLAYER_1);
    REQUIRE(board.score() == 2.0f);
    REQUIRE(board.outcome(0) == 1.0f);
    REQUIRE(board.outcome(1) == 0.0f);
    REQUIRE(board.render() == ".-.-.\n|1|1|\n.-.-.");
}

TEST_CASE("Copy is independent and cheap to share geometry", "[boxes_game]") {
    BoxesBoard board(2);
    BoxesBoard other = board;
    other.play_edge(0);
    REQUIRE(board.is_legal_edge(0));
    REQUIRE_FALSE(other.is_legal_edge(0));
    REQUIRE(&board.geometry() == &other.geometry());
}

TEST_CASE("Perft matches the hand-counted values", "[boxes_game]") {
    REQUIRE(boxes_perft(BoxesBoard(1, 1), 4) == std::make_tuple(24LL, 0LL, 24LL));
    REQUIRE(boxes_perft(BoxesBoard(1, 2), 4) == std::make_tuple(840LL, 0LL, 48LL));
    REQUIRE(boxes_perft(BoxesBoard(2, 2), 5) == std::make_tuple(95040LL, 3072LL, 768LL));
}
