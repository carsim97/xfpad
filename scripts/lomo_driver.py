"""Leave-one-material-out driver — batch runner for Phase 3a.

Enumerates every missing leave-one-material-out cell plus the ablation
controls and executes each one, with one worker per GPU pinned via
CUDA_VISIBLE_DEVICES.

Idempotent: a cell whose output JSON already exists is skipped, so the driver
can be re-launched after an interruption and resumes where it stopped.

Scope
-----
The driver enumerates the exhaustive leave-one-material-out matrix --- every
known material of both sensors, removed in turn --- and the confound controls:

  * matched-size random removal  (--ablate-random-n N)
  * non-anchor matched subset    (--ablate-subset-of S --subset-n N)

Usage
-----
    # everything, 10 seeds, both GPUs:
    python scripts/lomo_driver.py --num-runs 10 --gpus 0 1

    # only the LOMO cells (skip controls), single GPU:
    python scripts/lomo_driver.py --blocks lomo --gpus 0

    # dry-run: print the cell queue without launching anything:
    python scripts/lomo_driver.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "lomo"

sys.path.insert(0, str(REPO))
from scripts._protocol import AUDITED  # noqa: E402

# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------
# EXHAUSTIVE leave-one-material-out: every known PAI on both sensors, over 10
# seeds. Substrings follow the match rules in configs/<scanner>.yaml;
# material_name is the prototype name p_{u,k} is indexed by.
# (scanner, ablate substring, friendly name, material_name)
LOMO_CELLS = [
    # Green Bit — 7 known PAIs
    ("greenbit", "Latex",   "without_latex",       "Latex"),
    ("greenbit", "Fast",    "without_rprofast",    "RProFast"),
    ("greenbit", "LATEX",   "without_latex_v2",    "Latex V2"),
    ("greenbit", "RPRO10",  "without_rpro10",      "RPro10"),
    ("greenbit", "Ecoflex", "without_ecoflex",     "Ecoflex"),
    ("greenbit", "Body",    "without_body_double", "Body Double"),
    ("greenbit", "Wood",    "without_wood_glue",   "Wood Glue"),
    # Dermalog — 4 known PAIs
    ("dermalog", "Latex",   "without_latex",       "Latex"),
    ("dermalog", "Fast",    "without_rprofast",    "RProFast"),
    ("dermalog", "RPRO10",  "without_rpro10",      "RPro10"),
    ("dermalog", "LATEX",   "without_latex_v2",    "Latex V2"),
]

# ---------------------------------------------------------------------------
# Phase 3b — manifold-guided dataset composition
# ---------------------------------------------------------------------------
# Each sensor starts from a FIXED reduced vocabulary; the excluded known
# materials form the candidate pool that can be "added back". We train one cell
# per candidate (plus the reduced baseline) and let the five selection
# strategies (xfpad / raw_nc / random / matched_size / most_diff) pick among the
# SAME trained cells afterwards — so the strategies cost no extra GPU time.
#
# Recovery(u, candidate) = APCER_reduced(u) - APCER_reduced+candidate(u)
#
# Ablation substrings are the materials to REMOVE, per configs/<scanner>.yaml.
# The keep/candidate partition follows an OUTCOME-INDEPENDENT rule (sample
# representation), NOT the
# attribution scores, so the candidate set cannot have been chosen to flatter
# the manifold. Green Bit: keep the 4 fully-sampled base materials (750 each),
# candidates = the 3 under-sampled added materials (400 each). Operational story:
# "I have the well-collected materials; which under-sampled one should I acquire?"
# Substrings verified to isolate materials exactly (no cross-contamination).
PHASE4_SETUP = {
    # keep {Latex, RProFast, Latex V2, RPro10}; candidates {Ecoflex, Body Double, Wood Glue}
    "greenbit": {
        "removed_for_reduced": ["Ecoflex", "Body", "Wood"],
        "candidates": {                      # candidate -> substrings still removed
            "Ecoflex":     ["Body", "Wood"],
            "Body Double": ["Ecoflex", "Wood"],
            "Wood Glue":   ["Ecoflex", "Body"],
        },
    },
    # keep {RProFast, RPro10} (RPro family); candidates {Latex, Latex V2} (Latex family)
    "dermalog": {
        "removed_for_reduced": ["Latex", "LATEX"],
        "candidates": {
            "Latex":    ["LATEX"],
            "Latex V2": ["Latex"],
        },
    },
}


def _phase4_cells():
    """(scanner, ablate_substrings, cell_name) for the reduced baseline and each
    reduced+candidate configuration."""
    cells = []
    for scanner, spec in PHASE4_SETUP.items():
        cells.append((scanner, spec["removed_for_reduced"], "phase4_reduced"))
        for material, removed in spec["candidates"].items():
            slug = material.lower().replace(" ", "_").replace("'", "")
            cells.append((scanner, removed, f"phase4_add_{slug}"))
    return cells


# Confound controls. N values match the training-set sizes of
# the ablated anchors (Table I: Wood Glue 400, Latex 750).
CONTROL_CELLS = [
    ("greenbit", {"random_n": 400}, "random_n400"),   # matches Wood Glue size
    ("greenbit", {"random_n": 750}, "random_n750"),   # matches Latex size
    ("dermalog", {"random_n": 750}, "random_n750"),   # matches any Dermalog PAI
]
# The count confound is covered by the matched-size random removal above, the
# identity confound by the non-anchor cells of the sweep itself; a subset
# control would add nothing, since removing part of a material whose full
# removal is null cannot produce a larger effect.


def build_queue(blocks: list[str], num_runs: int) -> list[dict]:
    cells: list[dict] = []
    if "baseline" in blocks:
        # Reference cells: no ablation. Required by correlate_attr_apcer.py
        # to compute the Delta / pooled-sigma shifts.
        for scanner in ("greenbit", "dermalog"):
            for backbone in AUDITED:
                cells.append({
                    "scanner": scanner, "backbone": backbone,
                    "name": "baseline",
                    "args": ["--ablation-name", "baseline"],
                })
    if "lomo" in blocks:
        for scanner, substr, name, _material in LOMO_CELLS:
            for backbone in AUDITED:
                cells.append({
                    "scanner": scanner, "backbone": backbone, "name": name,
                    "args": ["--ablate", substr, "--ablation-name", name],
                })
    if "phase4" in blocks:
        for scanner, removed, name in _phase4_cells():
            for backbone in AUDITED:
                cells.append({
                    "scanner": scanner, "backbone": backbone, "name": name,
                    "args": ["--ablate", *removed, "--ablation-name", name],
                })
    if "controls" in blocks:
        for scanner, spec, name in CONTROL_CELLS:
            for backbone in AUDITED:
                extra: list[str] = ["--ablation-name", name]
                if "random_n" in spec:
                    extra += ["--ablate-random-n", str(spec["random_n"])]
                else:
                    extra += ["--ablate-subset-of", spec["subset_of"],
                              "--subset-n", str(spec["subset_n"])]
                cells.append({
                    "scanner": scanner, "backbone": backbone, "name": name,
                    "args": extra,
                })
    for c in cells:
        c["json"] = OUT / (f"phase3_{c['scanner']}_{c['backbone']}_"
                           f"{c['name']}.json")
        c["args"] += ["--num-runs", str(num_runs)]
    return cells


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

CONFIG_SUFFIX = ""   # set by --config-suffix; "_cons" selects the
                     # num_workers=0 variants of the configs.


def run_cell(cell: dict, gpu: int, py: str, log_dir: Path) -> int:
    # CFD-PAD has its own entry point: its three-term objective and its
    # channel-importance pass do not fit the --backbone dispatch. The two
    # workers write the same JSON layout into the same directory.
    cfd = cell["backbone"] == "cfd_pad"
    script = "phase3_cfdpad.py" if cfd else "phase3_audit_pad.py"
    cmd = [py, str(REPO / "scripts" / script),
           "-c", str(REPO / "configs" / f"{cell['scanner']}{CONFIG_SUFFIX}.yaml")]
    if not cfd:
        cmd += ["--backbone", cell["backbone"]]
    cmd += ["--action", "both", "--save-json", str(cell["json"])] + cell["args"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    log_path = log_dir / (cell["json"].stem + ".log")
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n[cell] gpu={gpu} {time.ctime()}\n[cmd] {' '.join(cmd)}\n")
        r = subprocess.run(cmd, cwd=str(REPO), env=env,
                           stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[exit] {r.returncode}\n")
    return r.returncode


def worker(q: "queue.Queue[dict]", gpu: int, py: str, log_dir: Path,
           failures: list) -> None:
    while True:
        try:
            cell = q.get_nowait()
        except queue.Empty:
            return
        tag = cell["json"].stem
        print(f"[gpu{gpu}] START {tag}", flush=True)
        rc = run_cell(cell, gpu, py, log_dir)
        print(f"[gpu{gpu}] {'DONE ' if rc == 0 else 'FAILED'} {tag}", flush=True)
        if rc != 0:
            failures.append(tag)
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="LOMO + controls batch driver.")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1],
                        help="GPU ids to use (one worker per GPU).")
    parser.add_argument("--blocks", nargs="+",
                        default=["baseline", "lomo", "controls"],
                        choices=["baseline", "lomo", "controls", "phase4"],
                        help="Which cell blocks to enqueue. 'phase4' is the "
                             "guided-composition experiment of Phase 3b and is "
                             "NOT in the default set: run it after the main "
                             "matrix, with --blocks phase4.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter for the phase3 subprocesses.")
    parser.add_argument("--config-suffix", default="",
                        help="Suffix for the scanner configs, e.g. '_cons' to "
                             "use the num_workers=0 variants (required on this "
                             "machine: multi-process DataLoaders freeze it).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    global CONFIG_SUFFIX
    CONFIG_SUFFIX = args.config_suffix

    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(exist_ok=True)

    cells = build_queue(args.blocks, args.num_runs)
    pending = [c for c in cells if not c["json"].exists()]
    print(f"cells total={len(cells)}  done={len(cells) - len(pending)}  "
          f"pending={len(pending)}")
    for c in pending:
        print(f"  - {c['json'].stem}")
    if args.dry_run or not pending:
        return

    q: "queue.Queue[dict]" = queue.Queue()
    for c in pending:
        q.put(c)

    failures: list = []
    threads = [threading.Thread(target=worker,
                                args=(q, gpu, args.python, log_dir, failures))
               for gpu in args.gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    hrs = (time.time() - t0) / 3600
    print(f"\n=== LOMO driver finished in {hrs:.1f} h — "
          f"{len(pending) - len(failures)} ok, {len(failures)} failed ===")
    for f in failures:
        print(f"  FAILED: {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
