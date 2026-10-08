"""Read-only summarizer for the archived N563 rig JSONL cells.

Extracts, per cell file: the header (pose_noise_cfg, candidate/baseline knobs), the paired
`compare` record (Welch p on coverage_cont, Fisher p on success, rig `keep`), and the
reg_err_xy p50/p90 on fixture_B for BOTH arms. No arithmetic on coverage or success --
everything printed here is read straight out of the rig's own records.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

D = "results/aegis_v2"


def load(path):
    hdr, eps, summ, comp = None, [], [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rec = r.get("record")
        if rec == "header":
            hdr = r
        elif rec == "compare":
            comp.append(r)
        elif rec in ("summary", "summary_mode"):
            summ.append(r)
        else:
            eps.append(r)
    return hdr, eps, summ, comp


def pctl(vals, q, scale=1.0):
    """Percentile over the FINITE entries only. Empty sample -> None (rig _pct guard, N562)."""
    v = [scale * x for x in vals if x is not None and np.isfinite(x)]
    return round(float(np.percentile(v, q)), 4) if v else None


def cell(tag):
    path = os.path.join(D, f"{tag}.jsonl")
    if not os.path.exists(path):
        return None
    hdr, eps, summ, comp = load(path)
    out = {"tag": tag, "n_ep": len(eps), "n_summ": len(summ), "n_cmp": len(comp)}
    out["hdr"] = {k: hdr.get(k) for k in
                  ("seeds_requested", "aegis_reg", "aegis_reg_half_m", "aegis_reg_n",
                   "aegis_reg_est", "pose_noise_cfg", "path_mode", "backend")}
    if hdr:
        out["hdr_pose_noise"] = hdr.get("pose_noise")
        out["hdr_flags"] = hdr.get("knob_flags") or hdr.get("env_flags")
    cmp = None
    for r in comp:
        if r["record"] == "compare":
            cmp = r
    out["cmp"] = cmp
    for r in summ:
        out.setdefault("summ_" + r["record"], r)
    # Arm identification is taken from the rig's OWN summary_mode records, whose `path` label
    # carries the active knob string ("trochoid" = candidate, "trochoid[AEGIS_REG_EST=0.0]" =
    # paired baseline). Episodes are emitted as two contiguous equal blocks in that same order;
    # the split is then VALIDATED against the compare record (see NOTE: in this rig's compare
    # record `a` is the BASELINE arm and `b` is the CANDIDATE arm).
    half = len(eps) // 2
    modes = [r.get("path_mode") for r in summ if r.get("record") == "summary_mode"]
    cand_lbl = modes[0] if modes else "cand"
    base_lbl = modes[1] if len(modes) > 1 else "base"
    arms = {("cand[" + cand_lbl + "]"): eps[:half], ("base[" + base_lbl + "]"): eps[half:]}
    chk = {}
    for _name, es in arms.items():
        for su in ("fixture_A", "fixture_B", "fixture_R"):
            sel = [e for e in es if e["suite"] == su]
            chk.setdefault(su, []).append(sum(1 for e in sel if e.get("success")))
    valid_split = None
    if cmp is not None:
        valid_split = all(chk[su][0] == cmp[su]["succ_b"] and chk[su][1] == cmp[su]["succ_a"]
                          for su in chk)
    out["arm_split_matches_compare_record"] = valid_split
    by = defaultdict(list)
    for name, es in arms.items():
        for e in es:
            by[(name, e["suite"])].append(e)
    out["arms"] = sorted({k[0] for k in by})
    per = {}
    for (arm, suite), es in sorted(by.items()):
        per[f"{arm}/{suite}"] = {
            "n": len(es),
            "succ": sum(1 for e in es if e.get("success")),
            "covc_mean": round(float(np.mean([e["coverage_cont"] for e in es])), 4)
            if all(np.isfinite(e["coverage_cont"]) for e in es) else None,
            "reg_err_p50_mm": pctl([e.get("reg_err_xy_m") for e in es], 50, 1000.0),
            "reg_err_p90_mm": pctl([e.get("reg_err_xy_m") for e in es], 90, 1000.0),
            "reg_ok": sum(1 for e in es if e.get("reg_ok")),
            "escaped": sum(1 for e in es if e.get("escaped")),
            "border_p50": pctl([e.get("reg_border_frac") for e in es], 50),
            "gap_p50_mm": pctl([e.get("reg_est_gap_mm") for e in es], 50),
        }
    out["per"] = per
    return out


if __name__ == "__main__":
    tags = sys.argv[1:]
    if not tags:
        tags = sorted({f.split(".jsonl")[0] for f in os.listdir(D)
                       if f.startswith("N563") and f.endswith(".jsonl")})
    for t in tags:
        c = cell(t)
        if not c:
            print(f"### {t}: MISSING")
            continue
        print(f"### {t}  eps={c['n_ep']} summaries={c['n_summ']} compares={c['n_cmp']}")
        print("   hdr:", json.dumps(c["hdr"]))
        if c["hdr_flags"]:
            print("   flags:", json.dumps(c["hdr_flags"])[:400])
        cmp = c.get("cmp")
        if cmp:
            print("   compare: metric=", cmp.get("metric"), " keep=", cmp.get("keep"),
                  " pose_noise_cfg=", cmp.get("pose_noise_cfg"))
            for s in ("fixture_A", "fixture_B", "fixture_R"):
                if s in cmp:
                    print(f"     {s}:", json.dumps(cmp[s]))
            print("     keep_rule:", json.dumps(cmp.get("keep_rule")))
        print("   arm_split_ok:", c["arm_split_matches_compare_record"])
        for k, v in sorted(c["per"].items()):
            print(f"   {k}: {json.dumps(v)}")
        print()
