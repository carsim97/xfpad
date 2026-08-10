"""Is the CosFace prototype layout reproducible across seeds?

X-FPAD places the K prototypes at phi_k = 2*pi*k/K + theta_offset, with a single
shared learnable offset: the material -> sector map is deterministic and the only
freedom is a global rotation. CosFace learns K free prototypes, so it may reach a
near-uniform spacing yet put the materials in a different cyclic order every time
it is retrained. That is the difference that matters for an auditing tool: an
auditor who retrains the encoder must find the same compass.

Three measurements, all read straight from the checkpoints' stored prototypes:

  * spacing      -- adjacent angular separations vs the ideal 360/K;
  * cyclic order -- the sequence of materials around the circle, compared with
                    seed 0 up to rotation (and, optionally, reflection);
  * residual     -- after aligning each seed to seed 0 with the best RIGID
                    rotation, how far the materials still are from where seed 0
                    put them. Zero would mean "same layout, just rotated".

Usage
-----
    python scripts/cosface_layout_stability.py
    python scripts/cosface_layout_stability.py --variant arcface
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts._common import bona_fide_label, known_names  # noqa: E402
from xfpad.config import load_config  # noqa: E402


def _wrap(deg: np.ndarray) -> np.ndarray:
    """Wrap to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _circmean(deg: np.ndarray) -> float:
    r = np.radians(deg)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def load_layouts(scanner: str, variant: str):
    """[(seed, {material: angle_deg})] from the stored prototypes."""
    cfg = load_config(str(REPO / "configs" / f"{scanner}.yaml"))
    names = known_names(cfg)
    bf = bona_fide_label(names)
    # prototype k corresponds to the k-th PAI label in ascending label order
    order = [names[l] for l in sorted(names) if l != bf]

    out = []
    for ck in sorted(glob.glob(str(REPO / "checkpoints" /
                                   f"geometric_{scanner}_{variant}_[0-9].pth"))):
        d = torch.load(ck, map_location="cpu")
        P = np.asarray(d["prototypes"], dtype=float)
        assert P.shape == (len(order), 2), (
            f"{Path(ck).name}: prototypes {P.shape}, expected ({len(order)}, 2)")
        ang = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360.0
        out.append((int(d["seed"]), dict(zip(order, ang))))
    return order, out


def cyclic_sequence(layout: Dict[str, float], order: List[str]) -> List[str]:
    return [m for m in sorted(order, key=lambda m: layout[m])]


def same_cycle(a: List[str], b: List[str], allow_reflection: bool) -> bool:
    """Equal as a cyclic sequence, up to rotation (and optionally reflection)."""
    n = len(a)
    doubled = b + b
    for i in range(n):
        if doubled[i:i + n] == a:
            return True
    if allow_reflection:
        rev = list(reversed(b))
        doubled = rev + rev
        for i in range(n):
            if doubled[i:i + n] == a:
                return True
    return False


def rigid_residual(ref: Dict[str, float], other: Dict[str, float],
                   order: List[str]) -> float:
    """Mean |angular error| after the best global rotation aligning `other` to `ref`.

    Materials are matched by identity, not by position: the question is whether
    the SAME material ends up in the SAME place once the global rotation is
    removed.
    """
    diff = np.array([other[m] - ref[m] for m in order])
    theta = _circmean(diff)
    return float(np.abs(_wrap(diff - theta)).mean())


def measure(scanner: str, variant: str = "cosface",
            allow_reflection: bool = False) -> dict | None:
    """The three measurements, as a dictionary.

    Both the printout and the selftest read from here, so the numbers checked
    and the numbers reported are the same rather than two parallel
    computations.
    """
    order, layouts = load_layouts(scanner, variant)
    if not layouts:
        return None
    K = len(order)

    seps = []
    for _, lay in layouts:
        a = np.sort(np.array([lay[m] for m in order]))
        seps.append(np.diff(np.concatenate([a, [a[0] + 360.0]])))
    seps = np.concatenate(seps)

    ref_seed, ref = layouts[0]
    ref_cyc = cyclic_sequence(ref, order)
    same = sum(1 for _s, lay in layouts[1:]
               if same_cycle(ref_cyc, cyclic_sequence(lay, order), allow_reflection))
    reps: List[List[str]] = []
    for _, lay in layouts:
        c = cyclic_sequence(lay, order)
        if not any(same_cycle(c, r, allow_reflection) for r in reps):
            reps.append(c)
    res = [rigid_residual(ref, lay, order) for _s, lay in layouts[1:]]

    offs = []
    for ck in sorted(glob.glob(str(REPO / "checkpoints" / f"geometric_{scanner}_[0-9].pth"))):
        d = torch.load(ck, map_location="cpu")
        t = (d.get("angular") or {}).get("theta_offset")
        if t is not None:
            offs.append(float(np.degrees(np.asarray(t).reshape(-1)[0])))

    return {"K": K, "ideal": 360.0 / K, "n_seeds": len(layouts), "ref_seed": ref_seed,
            "spacing_mean": float(seps.mean()), "spacing_std": float(seps.std()),
            "same": same, "n_compared": len(layouts) - 1, "distinct": len(reps),
            "res_mean": float(np.mean(res)), "res_min": float(np.min(res)),
            "res_max": float(np.max(res)),
            "theta_mean": float(np.mean(offs)) if offs else None,
            "theta_std": float(np.std(offs)) if offs else None,
            "n_xfpad": len(offs)}


def selftest() -> None:
    """The values published in Section S7 of the supplementary."""
    gb = measure("greenbit")
    dm = measure("dermalog")
    assert gb and dm, "cosface checkpoints missing"
    for m, sc in ((gb, "greenbit"), (dm, "dermalog")):
        assert m["n_seeds"] == 10, (sc, m["n_seeds"])
        # the spacing is the uniform one: that is what makes the comparison
        # interesting, because the difference is not there
        assert abs(m["spacing_mean"] - m["ideal"]) < 0.5, (sc, m)
        # no seed recovers the material -> sector map of the first
        assert m["same"] == 0, (sc, m["same"])
        # and the residual exceeds half a sector, i.e. the misalignment is not
        # a global rotation but a reshuffle
        assert m["res_mean"] > m["ideal"] / 2, (sc, m)
        assert m["n_xfpad"] == 10 and abs(m["theta_mean"]) < 1.0, (sc, m)
    assert round(gb["res_mean"], 1) == 57.3 and round(dm["res_mean"], 1) == 55.1, (gb, dm)
    assert round(gb["spacing_mean"], 1) == 51.4 and round(dm["spacing_mean"], 1) == 90.0
    print(f"selftest ok — cosface: cyclic order recovered "
          f"{gb['same']}/{gb['n_compared']} and {dm['same']}/{dm['n_compared']}, "
          f"residual {gb['res_mean']:.1f}° and {dm['res_mean']:.1f}°")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scanners", nargs="+", default=["greenbit", "dermalog"])
    ap.add_argument("--variant", default="cosface", choices=["cosface", "arcface"])
    ap.add_argument("--allow-reflection", action="store_true",
                    help="Count a mirrored cyclic order as the same layout "
                         "(more generous to the baseline).")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    for scanner in args.scanners:
        m = measure(scanner, args.variant, args.allow_reflection)
        if m is None:
            print(f"[{scanner}] nessun checkpoint {args.variant}")
            continue
        print("=" * 92)
        print(f"{scanner.upper()} — {args.variant}, K = {m['K']}, {m['n_seeds']} seed")
        print("=" * 92)
        print(f"  spaziatura fra prototipi adiacenti: {m['spacing_mean']:.1f}° "
              f"± {m['spacing_std']:.1f}°   (uniforme ideale {m['ideal']:.1f}°)")
        print(f"  cyclic order equal to seed {m['ref_seed']}: "
              f"**{m['same']}/{m['n_compared']}**"
              f"   (up to rotation{' and reflection' if args.allow_reflection else ''})")
        print(f"  distinct layouts: {m['distinct']} of {m['n_seeds']} seeds")
        print(f"  residual after removing the rigid rotation: "
              f"**{m['res_mean']:.1f}°** mean (min {m['res_min']:.1f}, "
              f"max {m['res_max']:.1f})")
        print(f"    reference: a residual of {m['ideal'] / 2:.1f}° is already half a "
              f"sector; {m['ideal']:.1f}° is a whole one")
        if m["theta_mean"] is not None:
            print(f"  -> X-FPAD, same {m['n_xfpad']} seeds: material -> sector map "
                  f"deterministic, residual **0.0° by construction**; the only freedom is "
                  f"theta_offset = {m['theta_mean']:+.2f}° ± {m['theta_std']:.2f}°")
        print()


if __name__ == "__main__":
    main()
