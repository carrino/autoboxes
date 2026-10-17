"""Build a scratch-only solver profiler without changing search decisions or budgets."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sysconfig
from pathlib import Path

import torch


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) == 1, (path, old)
    path.write_text(text.replace(old, new))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpp = output / "cpp"
    shutil.copytree("src/alpha_go/cpp", cpp, ignore=shutil.ignore_patterns("build"))
    header, impl = cpp / "boxes/boxes_solver.h", cpp / "boxes/boxes_solver.cpp"
    replace(header, "    uint64_t nodes() const { return nodes_; }",
            "    std::vector<uint64_t> profile() const { return profile_; }\n"
            "    uint64_t nodes() const { return nodes_; }")
    replace(header, "    uint64_t nodes_ = 0;",
            "    std::vector<uint64_t> profile_ = std::vector<uint64_t>(7, 0);\n"
            "    uint64_t nodes_ = 0;")
    text = impl.read_text()
    impl.write_text("#include <chrono>\n" + text)
    replace(impl, "int BoxesSolver::value(uint64_t mask) {\n    nodes_ = 0;",
            "int BoxesSolver::value(uint64_t mask) {\n"
            "    const auto started = std::chrono::steady_clock::now();\n    nodes_ = 0;")
    replace(impl, "    return mtdf(mask);\n}",
            "    const int result = mtdf(mask);\n"
            "    ++profile_[4]; profile_[5] += nodes_;\n"
            "    profile_[6] += std::chrono::duration_cast<std::chrono::nanoseconds>(\n"
            "        std::chrono::steady_clock::now() - started).count();\n"
            "    return result;\n}")
    signature = "std::optional<int> BoxesSolver::value_within(uint64_t mask, uint64_t max_nodes) {"
    replace(impl, signature,
            "std::optional<int> BoxesSolver::value_within(uint64_t mask, uint64_t max_nodes) {\n"
            "    const auto started = std::chrono::steady_clock::now();")
    replace(impl, "    const int v = mtdf(mask);\n    if (aborted_)",
            "    const int v = mtdf(mask);\n"
            "    ++profile_[0]; profile_[1] += nodes_; profile_[2] += aborted_;\n"
            "    profile_[3] += std::chrono::duration_cast<std::chrono::nanoseconds>(\n"
            "        std::chrono::steady_clock::now() - started).count();\n"
            "    if (aborted_)")
    bindings = cpp / "bindings/bindings.cpp"
    replace(bindings, '        .def("nodes", &alpha_go::BoxesSolver::nodes,',
            '        .def("profile", &alpha_go::BoxesSolver::profile)\n'
            '        .def("nodes", &alpha_go::BoxesSolver::nodes,')
    sources = [cpp / "bindings/bindings.cpp", cpp / "go/go_game.cpp", cpp / "mcts/mcts.cpp",
               *sorted((cpp / "boxes").glob("*.cpp"))]
    sources = [p for p in sources if not p.name.endswith("_test.cpp")]
    extension = output / ("alpha_go_cpp" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = ["g++", "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC", "-pthread",
               "-I" + sysconfig.get_path("include"),
               "-I" + str(Path(torch.__file__).parent / "include"), "-I" + str(cpp),
               *map(str, sources), "-o", str(extension)]
    subprocess.run(command, check=True)
    print(extension)


if __name__ == "__main__":
    main()
