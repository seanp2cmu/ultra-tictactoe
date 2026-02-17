#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "game/cpp/board.hpp"
#include "ai/endgame/cpp/dtw.hpp"

namespace py = pybind11;

PYBIND11_MODULE(uttt_cpp, m) {
    m.doc() = "C++ Board + DTW for Ultimate Tic-Tac-Toe";
    
    py::class_<uttt::Board>(m, "Board")
        .def(py::init<>())
        .def("clone", &uttt::Board::clone)
        .def("get_cell", &uttt::Board::get_cell)
        .def("set_cell", &uttt::Board::set_cell)
        .def("make_move", &uttt::Board::make_move, py::arg("r"), py::arg("c"), py::arg("validate") = true)
        .def("undo_move", [](uttt::Board& self, int r, int c, int prev_completed, int prev_winner, py::object prev_last_move) {
            if (prev_last_move.is_none()) {
                self.undo_move(r, c, prev_completed, prev_winner, -1, -1, false);
            } else {
                py::tuple lm = prev_last_move.cast<py::tuple>();
                self.undo_move(r, c, prev_completed, prev_winner, lm[0].cast<int>(), lm[1].cast<int>(), true);
            }
        })
        .def("_is_valid_move", &uttt::Board::_is_valid_move)
        .def("get_legal_moves", &uttt::Board::get_legal_moves)
        .def("get_sub_board", &uttt::Board::get_sub_board)
        .def("get_completed_state", &uttt::Board::get_completed_state)
        .def("get_sub_count_pair", &uttt::Board::get_sub_count_pair)
        .def("is_game_over", &uttt::Board::is_game_over)
        .def("count_playable_empty_cells", &uttt::Board::count_playable_empty_cells)
        .def("check_winner", &uttt::Board::check_winner)
        .def("get_completed_boards_2d", &uttt::Board::get_completed_boards_2d)
        .def("set_completed_boards_2d", &uttt::Board::set_completed_boards_2d)
        .def("to_array", &uttt::Board::to_array)
        .def_property("last_move",
            [](const uttt::Board& self) -> py::object {
                if (!self.has_last_move) return py::none();
                return py::make_tuple(self.last_move_r, self.last_move_c);
            },
            [](uttt::Board& self, py::object val) {
                if (val.is_none()) {
                    self.set_last_move_none();
                } else {
                    py::tuple t = val.cast<py::tuple>();
                    self.set_last_move(t[0].cast<int>(), t[1].cast<int>());
                }
            })
        .def_readwrite("current_player", &uttt::Board::current_player)
        .def_readwrite("winner", &uttt::Board::winner)
        .def_readwrite("completed_mask", &uttt::Board::completed_mask)
        .def_property("x_masks",
            [](const uttt::Board& self) { return std::vector<int>(self.x_masks.begin(), self.x_masks.end()); },
            [](uttt::Board& self, const std::vector<int>& v) { for (int i = 0; i < 9; i++) self.x_masks[i] = v[i]; })
        .def_property("o_masks",
            [](const uttt::Board& self) { return std::vector<int>(self.o_masks.begin(), self.o_masks.end()); },
            [](uttt::Board& self, const std::vector<int>& v) { for (int i = 0; i < 9; i++) self.o_masks[i] = v[i]; })
        .def_property("sub_counts",
            [](const uttt::Board& self) {
                std::vector<std::vector<int>> result;
                for (const auto& sc : self.sub_counts) {
                    result.push_back({sc[0], sc[1]});
                }
                return result;
            },
            [](uttt::Board& self, const std::vector<std::vector<int>>& v) {
                for (int i = 0; i < 9; i++) {
                    self.sub_counts[i] = {v[i][0], v[i][1]};
                }
            })
        .def_property("completed_boards",
            [](const uttt::Board& self) { return self.get_completed_boards_2d(); },
            [](uttt::Board& self, const std::vector<std::vector<int>>& v) { self.set_completed_boards_2d(v); });
    
    // DTW Calculator
    py::class_<uttt::DTWCalculator>(m, "DTWCalculator")
        .def(py::init<bool, int, int>(),
             py::arg("use_cache") = true,
             py::arg("endgame_threshold") = 15,
             py::arg("max_nodes") = 10000000)
        .def("is_endgame", &uttt::DTWCalculator::is_endgame)
        .def("calculate_dtw", [](uttt::DTWCalculator& self, uttt::Board& board) -> py::object {
            auto result = self.calculate_dtw(board);
            if (result.result == -2) {
                return py::none();
            }
            py::object best_move = py::none();
            if (result.best_move_r >= 0) {
                best_move = py::make_tuple(result.best_move_r, result.best_move_c);
            }
            return py::make_tuple(result.result, result.dtw, best_move);
        })
        .def("reset_search_stats", &uttt::DTWCalculator::reset_stats)
        .def("get_stats", [](uttt::DTWCalculator& self) {
            py::dict stats;
            stats["dtw_searches"] = self.total_searches;
            stats["dtw_nodes"] = self.total_nodes;
            stats["dtw_aborted"] = self.aborted_searches;
            stats["dtw_avg_nodes"] = self.total_searches > 0 ? 
                (double)self.total_nodes / self.total_searches : 0.0;
            stats["cache_size"] = self.cache.size();
            return stats;
        })
        .def_readwrite("endgame_threshold", &uttt::DTWCalculator::endgame_threshold)
        .def_readwrite("max_nodes", &uttt::DTWCalculator::max_nodes)
        .def_readwrite("use_cache", &uttt::DTWCalculator::use_cache);
}
