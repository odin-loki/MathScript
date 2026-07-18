"""Union-resolve git conflict markers. Help catalog lines merge APIs."""
import re
import sys
from pathlib import Path


def apis(s: str) -> set[str]:
    return set(
        re.findall(
            r"(?:graph_|geo_|crypto_|signal_|mpi_|stats_|prob_|info_|finance_|numthy_|combo_|im|hess|schur|sqrtm|logm|tril|triu)[a-z0-9_]*\([^)]*\)",
            s,
        )
    )


def resolve(path: str) -> None:
    p = Path(path)
    c = p.read_text(encoding="utf-8")
    pattern = r"<<<<<<< HEAD\r?\n([\s\S]*?)=======\r?\n([\s\S]*?)>>>>>>>[^\r\n]*\r?\n"
    matches = list(re.finditer(pattern, c))
    print(f"{path}: {len(matches)} conflicts")
    if not matches:
        return
    out = c
    for m in reversed(matches):
        h, t = m.group(1), m.group(2)
        helpish = (h.strip().startswith('"') or t.strip().startswith('"')) and any(
            k in h + t
            for k in (
                "graph_pagerank",
                "info_entropy",
                "stats_mean",
                "hess",
                "schur",
                "imgradient",
                "finance_efficient",
                "numthy_factor",
                "combo_binomial",
                "geo_bezier",
            )
        )
        if helpish:

            def score(s: str) -> int:
                keys = [
                    "hess",
                    "schur",
                    "imgradient_morph",
                    "finance_efficient_frontier",
                    "finance_max_sharpe",
                    "numthy_factor",
                    "combo_binomial",
                    "combo_bell_num",
                    "geo_bezier_eval",
                    "geo_catmull_rom",
                    "geo_bspline_eval",
                ]
                return sum(1 for k in keys if k in s)

            base = t if score(t) > score(h) else (h if score(h) > score(t) else (h if len(h) >= len(t) else t))
            other = t if base is h else h
            missing = apis(other) - apis(base)
            merged = base
            if missing:
                extra = ", ".join(sorted(missing))
                idx = merged.rfind('\\n"')
                if idx >= 0:
                    merged = merged[:idx] + ", " + extra + merged[idx:]
                print(f"  catalog: inserted {len(missing)} apis")
        elif "TEST(" in h or "TEST(" in t:
            merged = h + t
            if h.rstrip() and "TEST(" in t and not re.search(r"\}\s*$", h.rstrip()):
                if "EXPECT_" in h or "ASSERT_" in h:
                    merged = h.rstrip() + "\n}\n\n" + t.lstrip()
                    print("  added missing closing brace")
        else:
            merged = h + t
            print(f"  union code lenses {len(h)}+{len(t)}")
        out = out[: m.start()] + merged + out[m.end() :]
    if "<<<<<<<" in out:
        print("STILL CONFLICT", path)
        sys.exit(1)
    p.write_text(out, encoding="utf-8", newline="")
    print("ok", path)


for arg in sys.argv[1:]:
    resolve(arg)
