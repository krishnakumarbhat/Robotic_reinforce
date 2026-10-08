"""Purpose: readout for N480 (run 498) — decompose the compound pose-noise label.
Inputs: results/aegis_v2/N480_r498_*.jsonl. Outputs: the D1-D5 table on stdout.
Pure aggregation of the rig's own records; no arithmetic on coverage/success.
Arm split: the rig streams the 60 candidate episodes first, then the 60 --compare
baseline episodes (verified against the two `summary_mode` records in each file)."""
import glob
import json
import math
import statistics as st

CELLS = ["0.32,0", "0.96,0", "1.60,0", "2.56,0", "0,64", "0,192", "0,320", "0,640", "0,1280",
         "0.32,64", "1.60,320", "2.56,320"]
SUITES = ("fixture_A", "fixture_B", "fixture_R")


def load(cell):
    tag = cell.replace(",", "_").replace(".", "_")
    kind = "trans" if cell.endswith(",0") and not cell.startswith("0,") else "cell"
    hits = glob.glob(f"results/aegis_v2/N480_r498_{kind}{tag}.jsonl")
    assert len(hits) == 1, (cell, hits)
    return hits[0]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(int(p * (len(xs) - 1)), len(xs) - 1)] if xs else float("nan")


def agg(eps):
    d = dict(n=len(eps), succ=sum(1 for e in eps if e["success"]) / len(eps),
             cov=st.mean([e["coverage_cont"] for e in eps]),
             reg_ok=sum(1 for e in eps if e.get("reg_ok")) / len(eps),
             esc=sum(1 for e in eps if e.get("escaped")) / len(eps),
             k=sorted({e["reg_casts"] for e in eps}),
             casts=sum(e["reg_casts"] for e in eps),
             err50=q([e["reg_err_xy_m"] for e in eps if e["reg_err_xy_m"] == e["reg_err_xy_m"]], .5),
             err90=q([e["reg_err_xy_m"] for e in eps if e["reg_err_xy_m"] == e["reg_err_xy_m"]], .9),
             yaw90=q([e["reg_yaw_plan_deg"] for e in eps if e["reg_yaw_plan_deg"] == e["reg_yaw_plan_deg"]], .9),
             off95=q([math.hypot(e["pose_noise"][0], e["pose_noise"][1]) for e in eps], .95))
    return d


DATA = {}
for cell in CELLS:
    f = load(cell)
    hdr = cmp_ = None
    summ, eps = [], []
    for line in open(f, errors="ignore"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("record") == "header":
            hdr = r
        elif r.get("record") == "compare":
            cmp_ = r
        elif r.get("record") == "summary_mode":
            summ.append(r)
        elif r.get("record") == "episode":
            eps.append(r)
    half = len(eps) // 2
    cand = {s: agg([e for e in eps[:half] if e["suite"] == s]) for s in SUITES}
    base = {s: agg([e for e in eps[half:] if e["suite"] == s]) for s in SUITES}
    # cross-check the hand-rolled split against the rig's own summaries
    for arm, s_rec in ((cand, summ[0]), (base, summ[1])):
        for s in SUITES:
            assert abs(arm[s]["succ"] - s_rec["per_suite"][s]["transfer_success"]) < 1e-9, (cell, arm is cand, s)
            # the rig's own summary rounds coverage_cont to 4 dp
            assert abs(arm[s]["cov"] - s_rec["per_suite"][s]["mean_coverage_cont"]) < 1e-4, (cell, arm is cand, s)
    DATA[cell] = dict(cand=cand, base=base, cmp=cmp_, hdr=hdr,
                      verdict_c=summ[0]["verdict"], verdict_b=summ[1]["verdict"],
                      harness=summ[0]["harness_errors"] + summ[1]["harness_errors"])

hdr = "cell      arm    | succ   cov     regok  esc    k          casts   err_p50  err_p90  yawp90  off_p95"
print(hdr)
print("-" * len(hdr))
for cell in CELLS:
    for arm in ("cand", "base"):
        B = DATA[cell][arm]["fixture_B"]
        print(f"{cell:9s} {arm:6s} | {B['succ']:.2f}  {B['cov']:.4f}  {B['reg_ok']:.2f}  {B['esc']:.2f}  "
              f"{str(B['k']):10s} {B['casts']:6d}  {B['err50']*1000:6.2f}mm {B['err90']*1000:6.2f}mm "
              f"{B['yaw90']:6.2f}  {B['off95']:6.3f}m")

print("\n=== PRIMARY (fixture_B) paired certificate, candidate lattice vs frozen SINGLE-CAST champion ===")
print("cell        | cand_succ base_succ  dCov     Welch_p   Fisher_p  rig_keep  verdict(cand/base)  harness")
for cell in CELLS:
    b = DATA[cell]["cmp"]["fixture_B"]
    D = DATA[cell]
    print(f"{cell:10s} | {b['succ_b']/b['n_b']:9.2f} {b['succ_a']/b['n_a']:9.2f}  {b['mean_b']-b['mean_a']:+8.4f}  "
          f"{b['welch_p']:9.2e}  {b['fisher_p']:9.2e}  {str(D['cmp']['keep']):8s}  "
          f"{D['verdict_c']}/{D['verdict_b']}  {D['harness']}")

print("\n=== D1: compound vs translation-only at the SAME sigma_t (fixture_B, candidate arm, same 20 seeds) ===")
for comp, tr in (("0.32,64", "0.32,0"), ("1.60,320", "1.60,0"), ("2.56,320", "2.56,0")):
    a, b = DATA[comp]["cand"]["fixture_B"], DATA[tr]["cand"]["fixture_B"]
    print(f"  {comp:9s} vs {tr:9s}: succ {a['succ']:.2f} vs {b['succ']:.2f} | cov {a['cov']:.4f} vs {b['cov']:.4f} "
          f"| dcov {a['cov']-b['cov']:+.4f} | yaw_p90 {a['yaw90']:.2f} vs {b['yaw90']:.2f} | casts {a['casts']} vs {b['casts']}")

print("\n=== D2/D3: yaw-only (sigma_t = 0 -> k = 1, the lattice is inert; candidate == single-cast) ===")
print("cell      | B succ  B cov    B regok || A succ  A cov    A regok || R succ  R cov")
for cell in ("0,64", "0,192", "0,320", "0,640", "0,1280"):
    c = DATA[cell]["cand"]
    print(f"{cell:9s} | {c['fixture_B']['succ']:6.2f}  {c['fixture_B']['cov']:.4f}  {c['fixture_B']['reg_ok']:6.2f} || "
          f"{c['fixture_A']['succ']:6.2f}  {c['fixture_A']['cov']:.4f}  {c['fixture_A']['reg_ok']:6.2f} || "
          f"{c['fixture_R']['succ']:6.2f}  {c['fixture_R']['cov']:.4f}")

print("\n=== D4: ray budget (sum reg_casts over the 20 candidate fixture_B episodes) ===")
for cell in CELLS:
    print(f"  {cell:9s} k={DATA[cell]['cand']['fixture_B']['k']}  casts={DATA[cell]['cand']['fixture_B']['casts']}")
print("\nrig md5 must be ce401a0293b695b7b56c066024c1003f (0 bytes changed); "
      f"tool teleport / coverage arithmetic: none (no rig byte touched)")
