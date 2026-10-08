"""Purpose: read the N190 paired runs (one compare record each) into a single table.
Inputs: file globs. Outputs: printed table (candidate vs paired baseline, per suite).
"""
import glob
import json
import sys

import numpy as np

for p in sorted(sys.argv[1:]):
    rows = [json.loads(x) for x in open(p)]
    h, c = rows[0], [r for r in rows if r.get("record") == "compare"][0]
    cb, bb = c["candidate_knobs"], c["baseline_knobs"]
    print(f"== {p.split('/')[-1]}  noise={h['pose_noise_cfg']} "
          f"cand(half={cb['AEGIS_REG_HALF_M']},est={cb['AEGIS_REG_EST'] or 'mean'}) "
          f"base(half={bb['AEGIS_REG_HALF_M']},est={bb['AEGIS_REG_EST'] or 'mean'}) "
          f"keep={c.get('keep')}")
    for s in ("fixture_A", "fixture_B", "fixture_R"):
        d = c[s]
        print(f"   {s}: covc {d['mean_a']:.4f}->{d['mean_b']:.4f} d={d['delta']:+.4f} "
              f"welch_p={d['welch_p']:.3g} paired_p={d['paired_p']:.3g} "
              f"succ {d['succ_a']}->{d['succ_b']} fisher_p={d['fisher_p']:.3g}")
    eps = [r for r in rows if r.get("record") == "episode"]
    for arm, sl in (("cand", eps[:len(eps) // 2]), ("base", eps[len(eps) // 2:])):
        e = np.array([r["reg_err_xy_m"] for r in sl if r.get("reg_err_xy_m") is not None])
        print(f"   {arm}: reg_err med={np.median(e) * 1000:7.2f}mm "
              f"p90={np.percentile(e, 90) * 1000:7.2f}mm max={e.max() * 1000:7.2f}mm "
              f"reg_ok={int(sum(1 for r in sl if r.get('reg_ok')))}/{len(sl)} "
              f"harness={sum(1 for r in sl if r['backend'] == 'none')}")
