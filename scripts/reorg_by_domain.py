#!/usr/bin/env python3
"""Regroup tests and matrix-call handlers by mathematical domain, not wave number."""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "tests" / "unit"
INTEGRATION = ROOT / "tests" / "integration"
NUMERICAL = ROOT / "tests" / "numerical"
PERF = ROOT / "tests" / "performance"
MATRIX_CALLS = ROOT / "src" / "interp" / "matrix_calls"

PREFIX_DOMAINS: list[tuple[str, str]] = [
    ("cellmemory_", "frameworks"),
    ("cellai_", "frameworks"),
    ("tensorops_", "tensorops"),
    ("diffgeo_", "diffgeo"),
    ("quantum_", "quantum"),
    ("control_", "control"),
    ("signal_", "signal"),
    ("stats_", "stats"),
    ("crypto_", "crypto"),
    ("compress_", "compress"),
    ("finance_", "finance"),
    ("numthy_", "numthy"),
    ("combo_", "combo"),
    ("sparse_", "core"),
    ("info_", "info"),
    ("cplx_", "cplx"),
    ("poly_", "poly"),
    ("prob_", "prob"),
    ("graph_", "graph"),
    ("topo_", "topo"),
    ("geo_", "geo"),
    ("fem_", "fem"),
    ("cfd_", "cfd"),
    ("ml_", "ml"),
    ("ode_", "ode"),
    ("pde_", "pde"),
    ("fft_", "fft"),
    ("dist_", "distributed"),
    ("cuda_", "cuda"),
    ("simd_", "simd"),
    ("special_", "special"),
    ("symbolic_", "symbolic"),
    ("optim_", "optim"),
    ("image_", "image"),
    ("gria_", "frameworks"),
    ("izaac_", "frameworks"),
    ("axiom_", "frameworks"),
    ("cypha_", "frameworks"),
    ("bignum_", "bignum"),
]

LINALG = {
    "eye", "zeros", "ones", "rand", "randn", "diag", "tril", "triu", "transpose",
    "repmat", "linspace", "logspace", "solve", "lu", "qr", "chol", "ldl", "eig",
    "svd", "schur", "hessenberg", "expm", "logm", "sqrtm", "sinm", "cosm", "funm",
    "pinv", "null", "orth", "kron", "bidiag", "bicgstab", "cg", "gmres", "minres",
    "qmr", "tfqmr", "lsmr", "lsqr", "jacobi", "precond_diag", "inv", "rref",
    "solve_sylvester", "hess", "balance", "ordschur", "qz", "gees", "lu_solve",
    "chol_solve", "matmul", "norm", "cond", "rank", "trace", "det", "hadamard",
    "householder", "givens", "companion", "toeplitz", "hankel", "vandermonde",
    "hilbert", "pascal", "magic", "vander", "blkdiag", "kronecker", "pinv",
}

IMAGE = {
    "canny", "sobel", "sobel_x", "sobel_y", "scharr", "roberts", "rgb2gray",
    "rgb2hsv", "bilateral", "boxfilter", "watershed", "slic", "radon", "sharpen",
    "shi_tomasi", "threshold_otsu", "threshold_binary", "adapthisteq", "histeq",
    "histogram", "resize", "rotate", "gaussian_blur", "median_blur", "erode",
    "dilate", "morph_open", "morph_close", "laplacian", "prewitt", "hog", "lbp",
    "integral_image", "clahe", "nlm_denoise", "unsharp",
}

FFT = {
    "fft", "ifft", "rfft", "irfft", "rfftfreq", "fftfreq", "fftshift", "ifftshift",
    "fft2", "ifft2", "dct", "idct", "dst", "idst", "dft", "dft_magnitude",
    "fft_dct2", "fft_dft", "fft_dst2",
}

PDE = {
    "heat", "heat1d", "heat2d", "wave1d", "wave2d", "poisson", "poisson1d",
    "poisson2d", "advection", "advection1d", "helmholtz",
}

ODE = {"rk4", "rk45", "ode45", "ode15s", "euler", "midpoint"}

COMPRESS = {
    "bwt_encode_vec", "bwt_decode_vec", "rle_encode_vec", "rle_decode_vec",
    "bzip2_compress_vec", "bzip2_decompress_vec", "wavelet_compress_vec",
    "wavelet_decompress_vec", "ans_encode_vec", "ans_decode_vec",
    "arithmetic_encode_vec", "arithmetic_decode_vec", "delta_encode_vec",
    "delta_decode_vec", "run_length_encode_vec", "run_length_decode_vec",
    "huffman_encode_vec", "huffman_decode_vec", "lz4_compress_vec",
    "lz4_decompress_vec",
}

FINANCE = {"simulate_gbm_path", "run_backtest", "run_backtest_equity"}

SPECIAL = {
    "sph_harm", "bessel_j", "bessel_y", "bessel_i", "bessel_k", "bessel_h",
    "legendre_p", "legendre_q", "chebyshev_t", "chebyshev_u", "hermite_h",
    "jacobi_p", "jacobi_am", "jacobi_sn", "jacobi_cn", "jacobi_dn", "jacobi_sc",
    "jacobi_sd", "jacobi_ds", "gamma", "erfinv", "erfcinv", "zeta", "airya",
    "airyb", "struve_h", "struve_l",
}

FILE_PREFIX_DOMAINS: list[tuple[str, str]] = [
    ("test_repl_", "repl"),
    ("test_mathscriptc_", "interp"),
    ("test_server_", "interp"),
    ("test_plot_", "interp"),
    ("test_jit_", "interp"),
    ("test_dist_", "distributed"),
    ("test_mpi_", "distributed"),
    ("test_nccl_", "cuda"),
    ("test_cuda_", "cuda"),
    ("test_fft_", "fft"),
    ("test_signal_", "signal"),
    ("test_stats_", "stats"),
    ("test_prob_", "prob"),
    ("test_special_", "special"),
    ("test_symbolic_", "symbolic"),
    ("test_optim_", "optim"),
    ("test_linalg_", "linalg"),
    ("test_iterative_", "linalg"),
    ("test_blas_", "linalg"),
    ("test_float_linalg", "linalg"),
    ("test_sparse_", "core"),
    ("test_poly_", "poly"),
    ("test_ode_", "ode"),
    ("test_pde_", "pde"),
    ("test_fem", "fem"),
    ("test_cfd", "cfd"),
    ("test_ml", "ml"),
    ("test_image", "image"),
    ("test_compress", "compress"),
    ("test_crypto", "crypto"),
    ("test_graph", "graph"),
    ("test_geo", "geo"),
    ("test_diffgeo", "diffgeo"),
    ("test_topo", "topo"),
    ("test_tensorops", "tensorops"),
    ("test_tensor", "core"),
    ("test_control", "control"),
    ("test_quantum", "quantum"),
    ("test_cplx", "cplx"),
    ("test_finance", "finance"),
    ("test_info", "info"),
    ("test_combo", "combo"),
    ("test_numthy", "numthy"),
    ("test_bignum", "bignum"),
    ("test_frameworks", "frameworks"),
    ("test_axiom_", "frameworks"),
    ("test_cellai_", "frameworks"),
    ("test_gria_", "frameworks"),
    ("test_izaac_", "frameworks"),
    ("test_domain_", "domain"),
    ("test_simd", "simd"),
    ("test_runtime", "runtime"),
    ("test_dispatch", "runtime"),
    ("test_topology", "runtime"),
    ("test_load_balancer", "runtime"),
    ("test_thread_pool_", "runtime"),
    ("test_cpu_", "runtime"),
    ("test_memory", "core"),
    ("test_error_", "core"),
    ("test_rng", "core"),
    ("test_expr_", "core"),
    ("test_scalar", "core"),
    ("test_units", "core"),
    ("test_checked_", "core"),
    ("test_construction", "core"),
    ("test_core_", "core"),
    ("test_sym_", "core"),
    ("test_data_driven", "core"),
    ("test_gui_", "gui"),
    ("bench_fft", "fft"),
    ("bench_linalg", "linalg"),
    ("bench_matmul", "linalg"),
    ("bench_repl", "interp"),
    ("bench_special", "special"),
    ("bench_stats", "stats"),
    ("bench_rng", "core"),
    ("bench_simd", "simd"),
    ("bench_signal", "signal"),
    ("bench_ode", "ode"),
    ("bench_fem", "fem"),
    ("bench_optim", "optim"),
    ("bench_frameworks", "frameworks"),
    ("bench_tensorops", "tensorops"),
    ("bench_distributed", "distributed"),
    ("bench_poly", "poly"),
    ("bench_prob", "prob"),
    ("bench_crypto", "crypto"),
    ("bench_graph", "graph"),
    ("bench_topo", "topo"),
    ("bench_image", "image"),
    ("bench_geo", "geo"),
    ("bench_finance", "finance"),
    ("bench_quantum", "quantum"),
    ("bench_compress", "compress"),
]

LINALG_FILE = {
    "test_matmul", "test_lu", "test_qr", "test_solve", "test_norm", "test_transpose",
    "test_lu_typed", "test_solve_typed", "test_chol_typed", "test_linalg_ext",
    "test_linalg_chol", "test_linalg_edge", "test_linalg_decomp", "test_linalg_decomp_ext",
    "test_linalg_advanced", "test_linalg_iterative2", "test_integration",
}

CMD_RE = re.compile(
    r'(?:expect_ok|expect_contains|expect_error(?:_contains)?|interp\.execute)\(\s*'
    r'(?:interp\s*,\s*)?"([^"]*)"'
)
IDENT_CALL_RE = re.compile(r"\b([a-z][a-z0-9_]{2,})\s*\(")
TEST_RE = re.compile(r"TEST\s*\(\s*(\w+)\s*,\s*([A-Za-z_]\w*)\s*\)")
WAVE_TEST_RE = re.compile(r"^wave(\d+)_")
SKIP_IDENTS = {
    "expect_ok", "expect_contains", "expect_error", "expect_error_contains",
    "execute", "find", "has_value", "count", "at", "rows", "cols", "size",
    "static_cast", "dynamic_cast", "string", "vector", "optional", "near",
    "eq", "gt", "lt", "true", "false", "interp", "namespace", "using",
}


def classify_symbol(name: str) -> str:
    n = name.lower()
    for prefix, domain in PREFIX_DOMAINS:
        if n.startswith(prefix):
            return domain
    if n in LINALG or n.startswith("precond_"):
        return "linalg"
    if n in IMAGE:
        return "image"
    if n in FFT or n.startswith("fft") or n.startswith("ifft") or n.startswith("dct") or n.startswith("dst"):
        return "fft"
    if n in PDE or ((n.startswith("heat") or n.startswith("poisson") or n.startswith("wave")) and not n.startswith("wavelet")):
        return "pde"
    if n in ODE or n.startswith("rk"):
        return "ode"
    if n in COMPRESS or n.startswith("wavelet"):
        return "compress"
    if n in FINANCE:
        return "finance"
    if n in SPECIAL or n.startswith("bessel") or n.startswith("legendre") or n.startswith("chebyshev") or n.startswith("jacobi_"):
        return "special"
    if n.startswith("assemble_") or n.startswith("lagrange_"):
        return "fem"
    return "repl"


def classify_filename(stem: str) -> str:
    if stem in LINALG_FILE:
        return "linalg"
    for prefix, domain in FILE_PREFIX_DOMAINS:
        if stem.startswith(prefix):
            return domain
    if stem.startswith("test_"):
        rest = stem[5:]
        return classify_symbol(rest.split("_")[0]) if rest else "repl"
    if stem.startswith("bench_"):
        return classify_symbol(stem[6:])
    return classify_symbol(stem)


def classify_source_text(text: str) -> str:
    votes: dict[str, int] = defaultdict(int)
    for ident in IDENT_CALL_RE.findall(text):
        if ident in SKIP_IDENTS:
            continue
        domain = classify_symbol(ident)
        if domain != "repl":
            votes[domain] += 1
    if not votes:
        return "repl"
    return max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    parent = path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def unlink_retry(path: Path) -> None:
    import time
    last: Exception | None = None
    for _ in range(20):
        try:
            path.unlink()
            return
        except PermissionError as exc:
            last = exc
            time.sleep(0.2)
    if last:
        raise last


def move_retry(src: Path, dest: Path) -> None:
    import time
    last: Exception | None = None
    for _ in range(20):
        try:
            shutil.move(str(src), str(dest))
            return
        except PermissionError as exc:
            last = exc
            time.sleep(0.2)
    if last:
        raise last


def move_file(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = unique_path(dest)
    move_retry(src, dest)
    return dest


def rewrite_integration_names(text: str, domain: str) -> str:
    suite = "Integration" + domain.title().replace("_", "")
    text = re.sub(
        r"TEST\s*\(\s*ReplWave\d+Pipeline\s*,",
        f"TEST({suite}, ",
        text,
    )
    text = re.sub(
        r"TEST\s*\(\s*Wave\d+Pipeline\s*,",
        f"TEST({suite}, ",
        text,
    )
    text = re.sub(r"(?m)^// MathScript Integration Tests:.*\n(//.*\n)*", "", text, count=1)
    return text


def slug_from_integration(text: str, wave: str | None) -> str:
    cmds = CMD_RE.findall(text)
    callees: list[str] = []
    for cmd in cmds:
        m = re.search(r"=\s*([a-z][a-z0-9_]*)\s*\(", cmd)
        if m:
            callees.append(m.group(1))
        else:
            m = re.match(r"\s*([a-z][a-z0-9_]*)\s*\(", cmd)
            if m and m.group(1) not in {"help", "vars", "clear", "version"}:
                callees.append(m.group(1))
    seen: list[str] = []
    for c in callees:
        if c not in seen:
            seen.append(c)
        if len(seen) >= 2:
            break
    if seen:
        return "_".join(seen[:2])
    if wave:
        return f"wave{wave}"
    return "pipeline"


def command_fingerprint(text: str) -> str:
    cmds = tuple(CMD_RE.findall(text))
    if cmds:
        blob = "\n".join(cmds)
    else:
        blob = re.sub(r"\s+", " ", text)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def reorg_matrix_calls() -> int:
    moved = 0
    for src in sorted(MATRIX_CALLS.glob("*.cpp")):
        domain = classify_symbol(src.stem)
        dest = MATRIX_CALLS / domain / src.name
        if dest.resolve() == src.resolve():
            continue
        move_file(src, dest)
        moved += 1
    return moved


def reorg_flat_tests(folder: Path, prefix_keep: str | None = None) -> int:
    moved = 0
    for src in sorted(folder.glob("*.cpp")):
        if src.name == "CMakeLists.txt":
            continue
        domain = classify_filename(src.stem)
        dest = folder / domain / src.name
        move_file(src, dest)
        moved += 1
    return moved


def reorg_integration() -> tuple[int, int]:
    kept_fingerprints: dict[tuple[str, str], Path] = {}
    moved = 0
    dropped = 0
    for existing in INTEGRATION.glob("*/*.cpp"):
        try:
            text = existing.read_text(encoding="utf-8")
        except OSError:
            continue
        domain = existing.parent.name
        kept_fingerprints[(domain, command_fingerprint(text))] = existing
    for src in sorted(INTEGRATION.glob("*.cpp")):
        text = src.read_text(encoding="utf-8")
        wave_m = re.search(r"wave(\d+)", src.stem)
        domain = classify_source_text(text)
        if domain == "repl":
            domain = classify_filename(src.stem)
        fp = command_fingerprint(text)
        key = (domain, fp)
        is_wave = bool(wave_m) and "wave" in src.stem
        if is_wave and key in kept_fingerprints:
            unlink_retry(src)
            dropped += 1
            continue
        slug = slug_from_integration(text, wave_m.group(1) if wave_m else None)
        if is_wave:
            new_name = f"test_{slug}_pipeline.cpp"
            text = rewrite_integration_names(text, domain)
        else:
            new_name = src.name
        dest = INTEGRATION / domain / new_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest = unique_path(dest)
        if is_wave:
            dest.write_text(text, encoding="utf-8")
            unlink_retry(src)
        else:
            move_retry(src, dest)
        kept_fingerprints[key] = dest
        moved += 1
    return moved, dropped


def split_repl_command_tests() -> list[str]:
    helpers_src = UNIT / "repl_test_helpers.hpp"
    helpers_dest_dir = UNIT / "repl"
    helpers_dest_dir.mkdir(parents=True, exist_ok=True)
    if helpers_src.exists():
        shutil.move(str(helpers_src), str(helpers_dest_dir / "repl_test_helpers.hpp"))

    files = [UNIT / "test_repl_commands.cpp"]
    files.extend(sorted(UNIT.glob("test_repl_commands_w*.cpp")))
    tests_by_domain: dict[str, list[str]] = defaultdict(list)
    core_tests: list[str] = []
    used_names: set[tuple[str, str]] = set()
    seen_bodies: dict[tuple[str, str], str] = {}

    header = '''#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

'''

    def extract_tests(path: Path) -> list[tuple[str, str, str]]:
        text = path.read_text(encoding="utf-8")
        out: list[tuple[str, str, str]] = []
        for m in TEST_RE.finditer(text):
            start = m.start()
            brace = text.find("{", m.end())
            if brace < 0:
                continue
            depth = 0
            i = brace
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        body = text[start : i + 1]
                        out.append((m.group(1), m.group(2), body))
                        break
                i += 1
        return out

    for path in files:
        if not path.exists():
            continue
        for suite, name, body in extract_tests(path):
            clean = WAVE_TEST_RE.sub("", name)
            clean = re.sub(r"_tail\d+", "", clean)
            if not clean:
                clean = name
            domain = classify_source_text(body)
            if path.name == "test_repl_commands.cpp" and not WAVE_TEST_RE.match(name):
                domain = "repl"
            key = (suite, clean)
            norm = re.sub(r"\s+", " ", body)
            norm = WAVE_TEST_RE.sub("", norm)
            norm = re.sub(r"_tail\d+", "", norm)
            if key in used_names:
                if seen_bodies.get(key) == norm:
                    continue
                n = 2
                while (suite, f"{clean}_{n}") in used_names:
                    n += 1
                clean = f"{clean}_{n}"
                key = (suite, clean)
            used_names.add(key)
            seen_bodies[key] = norm
            body = TEST_RE.sub(lambda mm: f"TEST({mm.group(1)}, {clean})", body, count=1)
            if domain == "repl":
                core_tests.append(body)
            else:
                tests_by_domain[domain].append(body)
        if path.name != "test_repl_commands.cpp":
            path.unlink()

    core_path = UNIT / "repl" / "test_repl_commands.cpp"
    core_path.write_text(header + "\n\n".join(core_tests) + "\n", encoding="utf-8")
    if (UNIT / "test_repl_commands.cpp").exists():
        (UNIT / "test_repl_commands.cpp").unlink()

    sources = ["unit/repl/test_repl_commands.cpp"]
    for domain in sorted(tests_by_domain):
        dest = UNIT / "repl" / f"test_repl_commands_{domain}.cpp"
        dest.write_text(header + "\n\n".join(tests_by_domain[domain]) + "\n", encoding="utf-8")
        sources.append(f"unit/repl/{dest.name}")
    return sources


def write_interp_cmake() -> None:
    path = ROOT / "src" / "interp" / "CMakeLists.txt"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'file(GLOB _MS_MATRIX_CALL_SOURCES CONFIGURE_DEPENDS\n    "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls/*.cpp")',
        'file(GLOB_RECURSE _MS_MATRIX_CALL_SOURCES CONFIGURE_DEPENDS\n    "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls/*.cpp")',
    )
    text = text.replace(
        'file(GLOB _MS_MATRIX_CALL_STEMS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls"\n    "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls/*.cpp")',
        'file(GLOB_RECURSE _MS_MATRIX_CALL_STEMS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls"\n    "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls/*.cpp")',
    )
    path.write_text(text, encoding="utf-8")


def write_integration_cmake() -> None:
    (INTEGRATION / "CMakeLists.txt").write_text(
        """# Integration tests grouped by mathematical domain (see subdirectories).
file(GLOB_RECURSE _MS_INTEGRATION_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cpp")
list(SORT _MS_INTEGRATION_SOURCES)
foreach(_src ${_MS_INTEGRATION_SOURCES})
    get_filename_component(_stem ${_src} NAME_WE)
    get_filename_component(_dir ${_src} DIRECTORY)
    if(_dir STREQUAL "")
        set(_name ${_stem})
    else()
        string(REPLACE "/" "_" _dir ${_dir})
        string(REGEX REPLACE "^test_" "" _short ${_stem})
        set(_name "integration_${_dir}_${_short}")
    endif()
    add_ms_test(${_name} ${_src})
endforeach()
""",
        encoding="utf-8",
    )


def write_numerical_cmake() -> None:
    (NUMERICAL / "CMakeLists.txt").write_text(
        """file(GLOB_RECURSE _MS_NUMERICAL_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cpp")
list(SORT _MS_NUMERICAL_SOURCES)
foreach(_src ${_MS_NUMERICAL_SOURCES})
    get_filename_component(_stem ${_src} NAME_WE)
    string(REGEX REPLACE "^test_" "" _short ${_stem})
    add_ms_test(numerical_${_short} ${_src})
endforeach()
""",
        encoding="utf-8",
    )


def write_performance_cmake() -> None:
    extra = {
        "bench_repl": "ms_interp",
        "bench_special": "ms_special",
        "bench_special_memory": "ms_special",
        "bench_fem": "ms_fem",
        "bench_tensorops": "ms_tensorops",
        "bench_topo": "ms_topo",
    }
    lines = [
        """function(add_ms_bench name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mathscript benchmark::benchmark benchmark::benchmark_main)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )
endfunction()

file(GLOB_RECURSE _MS_BENCH_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "bench_*.cpp")
list(SORT _MS_BENCH_SOURCES)
foreach(_src ${_MS_BENCH_SOURCES})
    get_filename_component(_stem ${_src} NAME_WE)
    add_ms_bench(${_stem} ${_src})
endforeach()
"""
    ]
    for stem, lib in extra.items():
        lines.append(f"if(TARGET {stem})\n    target_link_libraries({stem} PRIVATE {lib})\nendif()\n")
    (PERF / "CMakeLists.txt").write_text("".join(lines), encoding="utf-8")


def write_tests_cmake(repl_sources: list[str]) -> None:
    repl_list = "\n    ".join(repl_sources)
    (ROOT / "tests" / "CMakeLists.txt").write_text(
        """include(CTest)

function(add_ms_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mathscript GTest::gtest_main)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    )
    if(MS_ENABLE_COVERAGE AND NOT MSVC)
        target_compile_options(${name} PRIVATE -fexceptions)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_executable(test_repl_commands
    @REPL_SOURCES@
)
target_link_libraries(test_repl_commands PRIVATE mathscript GTest::gtest_main)
target_include_directories(test_repl_commands PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/unit
)
if(MS_ENABLE_COVERAGE AND NOT MSVC)
    target_compile_options(test_repl_commands PRIVATE -fexceptions)
endif()
add_test(NAME test_repl_commands COMMAND test_repl_commands)
if(MSVC)
    target_compile_options(test_repl_commands PRIVATE /bigobj)
endif()

add_executable(test_mathscriptc_cli unit/interp/test_mathscriptc_cli.cpp)
target_link_libraries(test_mathscriptc_cli PRIVATE GTest::gtest_main)
target_compile_definitions(test_mathscriptc_cli PRIVATE "MATHSCRIPTC_PATH=\\"$<TARGET_FILE:mathscriptc>\\"")
add_dependencies(test_mathscriptc_cli mathscriptc)
add_test(NAME test_mathscriptc_cli COMMAND test_mathscriptc_cli)

add_executable(test_repl_cli unit/interp/test_repl_cli.cpp)
target_link_libraries(test_repl_cli PRIVATE GTest::gtest_main)
target_compile_definitions(test_repl_cli PRIVATE "MATHSCRIPT_REPL_PATH=\\"$<TARGET_FILE:mathscript-repl>\\"")
add_dependencies(test_repl_cli mathscript-repl)
add_test(NAME test_repl_cli COMMAND test_repl_cli)

if(TARGET mathscript-server)
    add_executable(test_server_cli unit/interp/test_server_cli.cpp)
    target_link_libraries(test_server_cli PRIVATE GTest::gtest_main)
    target_compile_definitions(test_server_cli PRIVATE "MATHSCRIPT_SERVER_PATH=\\"$<TARGET_FILE:mathscript-server>\\"")
    add_dependencies(test_server_cli mathscript-server)
    add_test(NAME test_server_cli COMMAND test_server_cli)
endif()

file(GLOB_RECURSE _MS_UNIT_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "unit/*.cpp")
list(SORT _MS_UNIT_SOURCES)
set(_MS_UNIT_SKIP
    unit/repl/test_repl_commands.cpp
    unit/interp/test_mathscriptc_cli.cpp
    unit/interp/test_repl_cli.cpp
    unit/interp/test_server_cli.cpp
    unit/gui/test_gui_text_transforms.cpp
)
foreach(_src ${_MS_UNIT_SOURCES})
    get_filename_component(_name ${_src} NAME)
    if(_name MATCHES "^test_repl_commands")
        continue()
    endif()
    list(FIND _MS_UNIT_SKIP ${_src} _idx)
    if(NOT _idx EQUAL -1)
        continue()
    endif()
    get_filename_component(_stem ${_src} NAME_WE)
    add_ms_test(${_stem} ${_src})
endforeach()

if(MS_BUILD_GUI)
    find_package(Qt6 QUIET COMPONENTS Core)
    if(Qt6_FOUND)
        add_executable(test_gui_text_transforms
            unit/gui/test_gui_text_transforms.cpp
            ${CMAKE_SOURCE_DIR}/src/gui/TextTransforms.cpp)
        target_link_libraries(test_gui_text_transforms PRIVATE Qt6::Core GTest::gtest_main)
        target_include_directories(test_gui_text_transforms PRIVATE ${CMAKE_SOURCE_DIR}/src)
        add_test(NAME test_gui_text_transforms COMMAND test_gui_text_transforms)
    endif()
endif()

add_subdirectory(compliance)
add_subdirectory(numerical)
add_subdirectory(integration)

if(MS_BUILD_FUZZ)
    add_subdirectory(fuzz)
endif()
""".replace("@REPL_SOURCES@", repl_list),
        encoding="utf-8",
    )


def main() -> int:
    print("matrix_calls...", flush=True)
    print(f"  moved {reorg_matrix_calls()}", flush=True)
    write_interp_cmake()

    print("unit (non-REPL)...", flush=True)
    special = {
        "test_repl_commands.cpp",
        "test_mathscriptc_cli.cpp",
        "test_repl_cli.cpp",
        "test_server_cli.cpp",
        "test_gui_text_transforms.cpp",
    }
    moved_unit = 0
    for src in sorted(UNIT.glob("test_*.cpp")):
        if src.name.startswith("test_repl_commands"):
            continue
        domain = classify_filename(src.stem)
        if src.name in {"test_mathscriptc_cli.cpp", "test_repl_cli.cpp", "test_server_cli.cpp"}:
            domain = "interp"
        elif src.name == "test_gui_text_transforms.cpp":
            domain = "gui"
        dest = UNIT / domain / src.name
        move_file(src, dest)
        moved_unit += 1
    print(f"  moved {moved_unit}", flush=True)

    print("REPL command tests...", flush=True)
    repl_dir = UNIT / "repl"
    existing_repl = sorted(repl_dir.glob("test_repl_commands*.cpp")) if repl_dir.exists() else []
    if existing_repl and not (UNIT / "test_repl_commands.cpp").exists():
        repl_sources = [f"unit/repl/{p.name}" for p in existing_repl]
        print(f"  already split ({len(repl_sources)} TUs)", flush=True)
    else:
        repl_sources = split_repl_command_tests()
        print(f"  TUs: {len(repl_sources)}", flush=True)

    print("numerical...", flush=True)
    print(f"  moved {reorg_flat_tests(NUMERICAL)}", flush=True)
    write_numerical_cmake()

    print("performance...", flush=True)
    print(f"  moved {reorg_flat_tests(PERF)}", flush=True)
    write_performance_cmake()

    print("integration...", flush=True)
    moved, dropped = reorg_integration()
    print(f"  kept {moved}, dropped duplicate wave pipelines {dropped}", flush=True)
    write_integration_cmake()

    write_tests_cmake(repl_sources)
    print("CMake rewritten.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
