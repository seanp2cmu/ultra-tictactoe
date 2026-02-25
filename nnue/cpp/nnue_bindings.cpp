#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "nnue_model.hpp"
#include "nnue_search.hpp"
#include "nnue_datagen.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nnue_cpp, m) {
    m.doc() = "C++ NNUE engine for Ultimate Tic-Tac-Toe";

    // Feature extraction (standalone function)
    m.def("extract_features", [](const uttt::Board& board) {
        int stm[nnue::NUM_FEATURES], nstm[nnue::NUM_FEATURES];
        int stm_n, nstm_n;
        nnue::extract_features(board, stm, stm_n, nstm, nstm_n);
        std::vector<int> stm_vec(stm, stm + stm_n);
        std::vector<int> nstm_vec(nstm, nstm + nstm_n);
        return py::make_tuple(stm_vec, nstm_vec);
    }, py::arg("board"),
       "Extract perspective features from a C++ Board. Returns (stm_indices, nstm_indices).");

    // NNUE Model
    py::class_<nnue::NNUEModel>(m, "NNUEModel")
        .def(py::init<>())
        .def("load", &nnue::NNUEModel::load, py::arg("path"),
             "Load weights from binary file.")
        .def("evaluate_board", &nnue::NNUEModel::evaluate_board, py::arg("board"),
             "Evaluate a Board position. Returns raw eval (unbounded).")
        .def("evaluate", [](const nnue::NNUEModel& self,
                            const std::vector<int>& stm,
                            const std::vector<int>& nstm) {
            return self.evaluate(stm.data(), static_cast<int>(stm.size()),
                                nstm.data(), static_cast<int>(nstm.size()));
        }, py::arg("stm_features"), py::arg("nstm_features"),
           "Evaluate from sparse feature index lists. Returns raw eval (unbounded).")
        .def_readonly("loaded", &nnue::NNUEModel::loaded)
        .def_readonly("accumulator_size", &nnue::NNUEModel::accumulator_size)
        .def_readonly("hidden1_size", &nnue::NNUEModel::hidden1_size)
        .def_readonly("hidden2_size", &nnue::NNUEModel::hidden2_size)
        .def_readonly("num_buckets", &nnue::NNUEModel::num_buckets)
        .def_readonly("bucket_divisor", &nnue::NNUEModel::bucket_divisor);

    // Search Result
    py::class_<nnue::SearchResult>(m, "SearchResult")
        .def_readonly("best_r", &nnue::SearchResult::best_r)
        .def_readonly("best_c", &nnue::SearchResult::best_c)
        .def_readonly("score", &nnue::SearchResult::score)
        .def_readonly("depth", &nnue::SearchResult::depth)
        .def_readonly("nodes", &nnue::SearchResult::nodes)
        .def_readonly("tt_hits", &nnue::SearchResult::tt_hits)
        .def_readonly("time_ms", &nnue::SearchResult::time_ms)
        .def("__repr__", [](const nnue::SearchResult& r) {
            return "SearchResult(move=(" + std::to_string(r.best_r) + "," +
                   std::to_string(r.best_c) + "), score=" +
                   std::to_string(r.score) + ", depth=" +
                   std::to_string(r.depth) + ", nodes=" +
                   std::to_string(r.nodes) + ", time=" +
                   std::to_string(r.time_ms) + "ms)";
        });

    // Search Engine
    py::class_<nnue::NNUESearchEngine>(m, "NNUESearchEngine")
        .def(py::init<nnue::NNUEModel*, int>(),
             py::arg("model"), py::arg("tt_size_mb") = 16,
             py::keep_alive<1, 2>())  // Engine keeps model alive
        .def("search", &nnue::NNUESearchEngine::search,
             py::arg("board"), py::arg("max_depth") = 8,
             py::arg("time_limit_ms") = 0,
             "Run iterative deepening search. Returns SearchResult.")
        .def("clear", &nnue::NNUESearchEngine::clear,
             "Clear TT, history, and killers.")
        .def("set_qsearch", &nnue::NNUESearchEngine::set_qsearch,
             py::arg("mode") = 2,
             "Set quiescence search mode: 0=off, 1=always on, 2=auto (mid-game only).")
        .def("get_qsearch", &nnue::NNUESearchEngine::get_qsearch);

    // ─── Data Generation ────────────────────────────────────────

    py::class_<nnue::DataGenConfig>(m, "DataGenConfig")
        .def(py::init<>())
        .def_readwrite("search_depth", &nnue::DataGenConfig::search_depth)
        .def_readwrite("qsearch_mode", &nnue::DataGenConfig::qsearch_mode)
        .def_readwrite("tt_size_mb", &nnue::DataGenConfig::tt_size_mb)
        .def_readwrite("lambda_search", &nnue::DataGenConfig::lambda_search)
        .def_readwrite("write_minply", &nnue::DataGenConfig::write_minply)
        .def_readwrite("write_maxply", &nnue::DataGenConfig::write_maxply)
        .def_readwrite("eval_limit", &nnue::DataGenConfig::eval_limit)
        .def_readwrite("random_skip_rate", &nnue::DataGenConfig::random_skip_rate)
        .def_readwrite("skip_noisy", &nnue::DataGenConfig::skip_noisy)
        .def_readwrite("skip_noisy_maxply", &nnue::DataGenConfig::skip_noisy_maxply)
        .def_readwrite("random_move_count", &nnue::DataGenConfig::random_move_count)
        .def_readwrite("random_move_temp", &nnue::DataGenConfig::random_move_temp)
        .def_readwrite("early_stop_empty", &nnue::DataGenConfig::early_stop_empty);

    // generate_data: returns (boards_numpy, values_numpy) directly
    m.def("generate_data", [](nnue::NNUEModel* model,
                               const nnue::DataGenConfig& config,
                               int num_games,
                               int num_threads,
                               uint64_t seed) {
        // Release GIL during C++ computation
        std::vector<nnue::TrainingSample> samples;
        {
            py::gil_scoped_release release;
            samples = nnue::generate_parallel(model, config, num_games, num_threads, seed);
        }

        size_t n = samples.size();
        // Create numpy arrays
        auto boards = py::array_t<int8_t>({(ssize_t)n, (ssize_t)92});
        auto values = py::array_t<float>({(ssize_t)n});

        auto b_ptr = boards.mutable_unchecked<2>();
        auto v_ptr = values.mutable_unchecked<1>();

        for (size_t i = 0; i < n; ++i) {
            for (int j = 0; j < 92; ++j)
                b_ptr(i, j) = samples[i].board[j];
            v_ptr(i) = samples[i].value;
        }

        return py::make_tuple(boards, values);
    },
    py::arg("model"),
    py::arg("config") = nnue::DataGenConfig{},
    py::arg("num_games") = 1000,
    py::arg("num_threads") = 4,
    py::arg("seed") = 42,
    "Generate training data via C++ NNUE self-play.\n"
    "Returns (boards: np.ndarray[N,92], values: np.ndarray[N]).");

    // batch_rescore: multi-threaded NNUE deep search rescore
    m.def("batch_rescore", [](nnue::NNUEModel* model,
                               py::array_t<int8_t> boards_arr,
                               int search_depth,
                               int num_threads,
                               int qsearch_mode,
                               int tt_size_mb) {
        auto buf = boards_arr.request();
        if (buf.ndim != 2 || buf.shape[1] != 92)
            throw std::runtime_error("boards must be (N, 92) int8 array");

        int num_positions = static_cast<int>(buf.shape[0]);
        const int8_t* data = static_cast<const int8_t*>(buf.ptr);

        std::vector<float> scores;
        {
            py::gil_scoped_release release;
            scores = nnue::batch_rescore(model, data, num_positions,
                                          search_depth, num_threads,
                                          qsearch_mode, tt_size_mb);
        }

        // Return as numpy array
        auto result = py::array_t<float>({(ssize_t)num_positions});
        auto r_ptr = result.mutable_unchecked<1>();
        for (ssize_t i = 0; i < num_positions; ++i)
            r_ptr(i) = scores[i];

        return result;
    },
    py::arg("model"),
    py::arg("boards"),
    py::arg("search_depth") = 10,
    py::arg("num_threads") = 16,
    py::arg("qsearch_mode") = 2,
    py::arg("tt_size_mb") = 32,
    "Batch rescore positions with NNUE deep search (multi-threaded C++).\n"
    "Args: model, boards(N,92 int8), search_depth, num_threads, qsearch_mode, tt_size_mb\n"
    "Returns: scores np.ndarray[N] (raw eval, unbounded).");

    // search_bench: single-position search returning detailed stats
    m.def("search_bench", [](nnue::NNUEModel* model,
                              py::array_t<int8_t> board_arr,
                              int search_depth,
                              int qsearch_mode,
                              int tt_size_mb) {
        auto buf = board_arr.request();
        if (buf.size != 92)
            throw std::runtime_error("board must be (92,) int8 array");

        const int8_t* arr = static_cast<const int8_t*>(buf.ptr);

        // Reconstruct board
        uttt::Board board;
        for (int sub = 0; sub < 9; ++sub) {
            int base_r = (sub / 3) * 3;
            int base_c = (sub % 3) * 3;
            for (int cell = 0; cell < 9; ++cell) {
                int r = base_r + cell / 3;
                int c = base_c + cell % 3;
                int8_t v = arr[r * 9 + c];
                if (v == 1) board.x_masks[sub] |= (1 << cell);
                else if (v == 2) board.o_masks[sub] |= (1 << cell);
            }
            board.sub_counts[sub] = {
                __builtin_popcount(board.x_masks[sub]),
                __builtin_popcount(board.o_masks[sub])
            };
        }
        for (int i = 0; i < 9; ++i) {
            board.completed_boards[i] = arr[81 + i];
            if (board.completed_boards[i] != 0)
                board.completed_mask |= (1 << i);
        }
        int active = arr[90];
        if (active >= 0) {
            board.has_last_move = true;
            board.last_move_r = active / 3;
            board.last_move_c = active % 3;
        }
        board.current_player = arr[91];
        board.check_winner();

        nnue::NNUESearchEngine engine(model, tt_size_mb);
        engine.set_qsearch(qsearch_mode);
        nnue::SearchResult result;
        {
            py::gil_scoped_release release;
            result = engine.search(board, search_depth);
        }

        py::dict d;
        d["score"] = result.score;
        d["depth"] = result.depth;
        d["nodes"] = result.nodes;
        d["tt_hits"] = result.tt_hits;
        d["time_ms"] = result.time_ms;
        d["best_r"] = result.best_r;
        d["best_c"] = result.best_c;
        return d;
    },
    py::arg("model"),
    py::arg("board"),
    py::arg("search_depth") = 8,
    py::arg("qsearch_mode") = 2,
    py::arg("tt_size_mb") = 16,
    "Search a single position, returning {score, depth, nodes, tt_hits, time_ms}.");
}
