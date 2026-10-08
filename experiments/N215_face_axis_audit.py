#!/usr/bin/env python3
"""N215 (run 364) audit: the FIXTURE-FACE axis, from the canonical rig's own JSONL.

Every number printed here is read out of results/aegis_v2/N215_r364_*.jsonl (PyBullet DIRECT,
system python3, local CPU). No arithmetic is applied to coverage_cont or success: this script only
aggregates fields the rig wrote, and the plan reaches come from the rig's OWN scrub_uv() (pure
geometry, no physics). Each claim group is an assert, so a broken ladder fails the script.

WHY THIS AXIS: in runs 1-363 the two top faces were bare literals (elongated box
halfExtents [0.34, 0.14, 0.12], round cylinder radius 0.32), `tank_shape` is a two-valued coin
aliased with the seed, and "Fixture-B zero-shot TRANSFER" is a claim about the face.

PRE-REGISTERED PREDICTIONS (equations.md ROW N215, stated before any run):
  F1 the across-patch (v) face margin is the binder: order of first loss v < u < round radius.
  F2 the cliff is at face_half = patch_half + r_eff (0.095-0.110 on v).
  F3 the binding channel is the CONTACT SET, not the escape test: coverage degrades monotonically
     while escaped stays 0.
  F4 raster and trochoid fail at the same face_v, within one ladder step.
  F5 the certificate is a region and the frozen face sits close to its edge.

Usage: python3 experiments/N215_face_axis_audit.py            (system python3)
"""
import glob
import json
import os
import statistics as st
import sys

RES = "results/aegis_v2"
RUN350 = f"{RES}/Run350_champion_health_r350.jsonl"
CLEAN = 0.90
FACE_FROZEN = {"elong_u": 0.34, "elong_v": 0.14, "round_r": 0.32}

# the ladders, in the order they were run (all trochoid / 20 seeds / pose noise 0,0 unless named)
V_LADDER = [("id", 0.140), ("fv_120", 0.120), ("fv_100", 0.100), ("fv_080", 0.080),
            ("fv_070", 0.070), ("fv_065", 0.065), ("fv_060", 0.060), ("fv_055", 0.055),
            ("fv_0525", 0.0525), ("fv_050", 0.050), ("fv_0475", 0.0475), ("fv_045", 0.045),
            ("fv_040", 0.040), ("fv_030", 0.030), ("fv_020", 0.020), ("fv_010", 0.010)]
U_LADDER = [("id", 0.340), ("fu_300", 0.300), ("fu_280", 0.280), ("fu_260", 0.260),
            ("fu_240", 0.240), ("fu_225", 0.225), ("fu_220", 0.220), ("fu_210", 0.210),
            ("fu_200", 0.200)]
R_LADDER = [("id", 0.320), ("fr_300", 0.300), ("fr_280", 0.280), ("fr_260", 0.260),
            ("fr_240", 0.240), ("fr_230", 0.230), ("fr_225", 0.225), ("fr_220", 0.220),
            ("fr_215", 0.215), ("fr_200", 0.200)]
RAS_V = [("ras_fv060", 0.060), ("ras_fv050", 0.050), ("ras_fv040", 0.040),
         ("ras_fv030", 0.030), ("ras_fv020", 0.020)]
NOISE_ARMS = [("id", "0,0"), ("tro_fv060n", "0.03,6"), ("tro_fv055n", "0.03,6")]


def load(name):
    """Candidate-arm episodes + compare + header of one N215 result file (arms are file halves)."""
    recs = [json.loads(line) for line in open(f"{RES}/N215_r364_{name}.jsonl") if line.strip()]
    eps = [r for r in recs if r.get("record") == "episode"]
    cmp_ = [r for r in recs if r.get("record") == "compare"][0]
    hdr = [r for r in recs if r.get("record") == "header"][0]
    half = len(eps) // 2
    return eps[:half], eps[half:], cmp_, hdr


def rows(eps, suite):
    return [e for e in eps if e["suite"] == suite]


def agg(eps, suite):
    r = rows(eps, suite)
    return {"cov": round(st.mean([e["coverage_cont"] for e in r]), 4),
            "succ": sum(1 for e in r if e["success"]),
            "esc": sum(1 for e in r if e.get("escaped")),
            "stall": round(st.mean([e.get("stall_frac", 0.0) for e in r]), 3),
            "launch": round(st.mean([e.get("launch_frac", 0.0) for e in r]), 4)}


def cand(name):
    """Purpose: the CANDIDATE half of one arm (arms are file halves: candidate, paired baseline).
    Inputs: arm name. Outputs: list of episode records.
    """
    if name not in CACHE:
        CACHE[name] = load(name)
    return CACHE[name][0]


def plan_reach(mode, shape):
    """Purpose: the COMMANDED plan's own extent per axis, from the rig's scrub_uv (no physics).
    Inputs: path mode, "elongated" | "round". Outputs: (reach_u, reach_v) in metres.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                    "benchmarks"))
    os.environ["AEGIS_POSE_NOISE"] = "0,0"
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location(
        "rig", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_aegis_sweep.py"))
    assert spec is not None and spec.loader is not None
    rig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rig)
    sp = {"tank_shape": shape, "surface": "matte", "offset_cm": 0, "angle_deg": 0}
    uv = rig.scrub_uv(sp, mode)
    us = [p[0] for p in uv]
    vs = [p[1] for p in uv]
    return max(abs(min(us)), abs(max(us))), max(abs(min(vs)), abs(max(vs)))


def cliff(ladder, suite, cached) -> tuple[float, float]:
    """Purpose: the (last_fail, last_pass] bracket of the face value at which `suite` stops
    certifying, i.e. the largest value reading coverage_cont == 1.0 AND success == n.
    Inputs: ladder [(arm, face_value)], suite name, cache. Outputs: (bracket, rows).
    """
    n = None
    table = []
    for name, face in ladder:
        if name not in cached:
            cached[name] = load(name)
        eps = cached[name][0]
        n = n or len(rows(eps, suite))
        c = agg(eps, suite)
        certified = abs(c["cov"] - 1.0) < 1e-9 and c["succ"] == n and c["esc"] == 0
        table.append((float(face), certified))
    passes = [f for f, ok in table if ok]
    assert passes, f"no certifying arm in the ladder for {suite}"
    best_pass = min(passes)                       # the SMALLEST face that still certifies
    below = [f for f, ok in table if (not ok) and f < best_pass]
    return (max(below) if below else -1.0, best_pass)


print("=" * 78)
print("N215 FIXTURE-FACE AXIS -- rig-side identity")
print("=" * 78)
CACHE: dict = {}
files = sorted(glob.glob(f"{RES}/N215_r364_*.jsonl"))
print(f"arm files: {len(files)}")
_id, _idb, _idc, _idh = load("id")
recs350 = [json.loads(x) for x in open(RUN350) if x.strip()]
eps350 = {(r["suite"], r["seed"]): r for r in recs350 if r.get("record") == "episode"}
idk = {(e["suite"], e["seed"]): e for e in _id}
IDENT = ("coverage_cont", "success", "escaped", "coverage", "stall_frac", "z_exc_max_m")
n_id = 0
for k in idk:
    if k in eps350:
        n_id += 1
        for f in IDENT:
            assert idk[k].get(f) == eps350[k].get(f), f"identity arm differs from run 350 on {f} {k}"
d_fn = max(abs(idk[k]["fn_mean"] - eps350[k]["fn_mean"]) for k in idk if k in eps350)
print(f"identity vs run 350: {n_id} episodes, bit-identical on {IDENT}, "
      f"max |d fn_mean| {d_fn:.4f} N (N210's cross-process force floor)")
print(f"identity header face_realized = {_idh['face_realized']} (frozen literals at default 0.0)")
assert _idh["face_hu_m"] == 0.0 and _idh["face_hv_m"] == 0.0 and _idh["face_r_m"] == 0.0
assert _idh["face_realized"]["elongated"] == [0.34, 0.14]
assert _idh["face_realized"]["round"] == [0.32, 0.32]
keeps = []
for f in files:
    recs = [json.loads(x) for x in open(f) if x.strip()]
    keeps.append([r for r in recs if r.get("record") == "compare"][0]["keep"])
assert not any(keeps), "a face arm claims keep: every arm must tie or regress vs the paired champion"
print(f"rig keep = false in all {len(keeps)} arms BY CONSTRUCTION (no keep claimed)")

REACH_T = plan_reach("trochoid", "elongated")
REACH_R = plan_reach("trochoid", "round")
REACH_RAS = plan_reach("raster", "elongated")
print(f"commanded plan reach, trochoid/elongated: u {REACH_T[0]:.4f} v {REACH_T[1]:.4f}")
print(f"commanded plan reach, trochoid/round:     u {REACH_R[0]:.4f} v {REACH_R[1]:.4f}")
print(f"commanded plan reach, raster/elongated:   u {REACH_RAS[0]:.4f} v {REACH_RAS[1]:.4f}")

print()
print("=" * 78)
print("N215.1  THE THREE LADDERS (candidate arm; the paired baseline reads 1.0000 / 20-20 everywhere)")
print("=" * 78)
print(f"{'arm':10s} {'face':>7s} | {'B cov/succ/esc':>22s} | {'A cov/succ/esc':>22s} | "
      f"{'R cov/succ/esc':>22s} | keep")
for name, face in V_LADDER:
    c, _, cm, _ = load(name)
    b, a, r = (agg(c, s_) for s_ in ("fixture_B", "fixture_A", "fixture_R"))
    print(f"{name:10s} v={face:<5.4f} | {b['cov']:.4f} {b['succ']:2d}/20 {b['esc']:2d} | "
          f"{a['cov']:.4f} {a['succ']:2d}/20 {a['esc']:2d} | {r['cov']:.4f} {r['succ']:2d}/20 "
          f"{r['esc']:2d} | {cm['keep']}")
for name, face in U_LADDER:
    c, _, cm, _ = load(name)
    b, a, r = (agg(c, s_) for s_ in ("fixture_B", "fixture_A", "fixture_R"))
    print(f"{name:10s} u={face:<5.4f} | {b['cov']:.4f} {b['succ']:2d}/20 {b['esc']:2d} | "
          f"{a['cov']:.4f} {a['succ']:2d}/20 {a['esc']:2d} | {r['cov']:.4f} {r['succ']:2d}/20 "
          f"{r['esc']:2d} | {cm['keep']}")
for name, face in R_LADDER:
    c, _, cm, _ = load(name)
    b, a, r = (agg(c, s_) for s_ in ("fixture_B", "fixture_A", "fixture_R"))
    print(f"{name:10s} r={face:<5.4f} | {b['cov']:.4f} {b['succ']:2d}/20 {b['esc']:2d} | "
          f"{a['cov']:.4f} {a['succ']:2d}/20 {a['esc']:2d} | {r['cov']:.4f} {r['succ']:2d}/20 "
          f"{r['esc']:2d} | {cm['keep']}")
for name, face in RAS_V:
    c, _, cm, _ = load(name)
    b = agg(c, "fixture_B")
    print(f"{name:10s} v={face:<5.4f} | RASTER B {b['cov']:.4f} {b['succ']:2d}/20 {b['esc']:2d} | "
          f"{cm['keep']}")

cl_v = cliff(V_LADDER, "fixture_B", CACHE)
cl_u = cliff(U_LADDER, "fixture_B", CACHE)
cl_r = cliff(R_LADDER, "fixture_A", CACHE)
cl_ras = cliff(RAS_V, "fixture_B", CACHE)
print()
print(f"fixture_B v cliff bracket (last_fail, last_pass] = {cl_v}   plan reach_v {REACH_T[1]:.4f}")
print(f"fixture_B u cliff bracket                    = {cl_u}   plan reach_u {REACH_T[0]:.4f}")
print(f"fixture_A r cliff bracket                    = {cl_r}   plan corner rho "
      f"{(REACH_R[0] ** 2 + REACH_R[1] ** 2) ** 0.5:.4f}")
print(f"fixture_B v cliff, RASTER                    = {cl_ras}   raster reach_v {REACH_RAS[1]:.4f}")

# --- F1: the v face is the first to bind, and it binds in FRACTIONAL terms ---------------
f_v = cl_v[1] / FACE_FROZEN["elong_v"]
f_u = cl_u[1] / FACE_FROZEN["elong_u"]
f_r = cl_r[1] / FACE_FROZEN["round_r"]
print()
print("=" * 78)
print("N215.2  F1 (CONFIRMED): order of first loss v < u < round, as fractions of the frozen face")
print("=" * 78)
print(f"v {cl_v[1]:.4f}/{FACE_FROZEN['elong_v']} = {f_v:.3f} | u {cl_u[1]:.4f}/"
      f"{FACE_FROZEN['elong_u']} = {f_u:.3f} | round r {cl_r[1]:.4f}/{FACE_FROZEN['round_r']} "
      f"= {f_r:.3f}")
assert f_v < f_u < f_r, f"F1 refuted: first-loss order {f_v:.3f} {f_u:.3f} {f_r:.3f}"

# --- F2 (REFUTED): the cliff is the plan's own reach, not patch_half + r_eff -------------
PATCH_HV, R_EFF = 0.050, (0.035, 0.040, 0.050)
pred_lo, pred_hi = PATCH_HV + min(R_EFF), PATCH_HV + max(R_EFF)
print()
print("=" * 78)
print("N215.3  F2 (REFUTED): the cliff is the PLAN REACH, 2.0x below patch_half + r_eff")
print("=" * 78)
print(f"pre-registered F2 cliff window = patch_v {PATCH_HV} + r_eff {R_EFF} = "
      f"[{pred_lo:.4f}, {pred_hi:.4f}]")
print(f"measured bracket = {cl_v}, and it CONTAINS plan reach_v {REACH_T[1]:.4f}")
assert cl_v[1] < pred_lo, f"F2 not refuted: measured cliff {cl_v[1]} inside the predicted window"
assert cl_v[0] < REACH_T[1] <= cl_v[1], f"plan reach {REACH_T[1]} not in bracket {cl_v}"
# the discriminator that separates "plan reach" from "patch half": raster's plan reaches 0.035,
# NOT the 0.050 patch, and raster's cliff moves with IT.
assert cl_ras[0] < REACH_RAS[1] <= cl_ras[1], f"raster cliff {cl_ras} misses its plan reach"
assert abs(REACH_T[1] - REACH_RAS[1]) > 0.010, "the two modes must have separated reaches"
print(f"raster's own reach {REACH_RAS[1]:.4f} (1.43x below trochoid's {REACH_T[1]:.4f}) and its "
      f"cliff {cl_ras} -> the law is the PLAN, not the patch")
assert cl_u[0] <= REACH_T[0] <= cl_u[1] + 0.005, f"u cliff {cl_u} misses plan reach_u"
# the ROUND face: the plan's extreme CORNER is what leaves the disc, and the cliff is a graded
# boundary (N215.7), so it is only bracketed -- no exact closed form is claimed for it (N213's
# lesson: never refit a scale to the data it was meant to predict).
RHO = (REACH_R[0] ** 2 + REACH_R[1] ** 2) ** 0.5
assert 0.0 < RHO - cl_r[1] <= 0.030, f"round cliff {cl_r} inconsistent with the plan corner {RHO}"
assert cl_r[1] < cl_r[0] + 0.011, "round cliff bracket should be a ladder step, not a wide band"

# --- F3 (REFUTED): the channel is contact loss -> slide-off -> escape, and it is a STEP ----
print()
print("=" * 78)
print("N215.4  F3 (REFUTED): escaped flips 0 -> 20/20 at the cliff; the loss is a step, not a ramp")
print("=" * 78)
for name, face in [("fv_0525", 0.0525), ("fv_0475", 0.0475), ("fv_045", 0.045), ("fv_030", 0.030)]:
    c = agg(cand(name), "fixture_B")
    print(f"  face_v {face:.4f}: cov {c['cov']:.4f} succ {c['succ']:2d}/20 esc {c['esc']:2d} "
          f"stall_frac {c['stall']:.3f} launch_frac {c['launch']:.4f}")
c_fail = agg(load("fv_0475")[0], "fixture_B")
c_pass = agg(load("fv_0525")[0], "fixture_B")
assert c_pass["esc"] == 0 and c_pass["cov"] == 1.0
assert c_fail["esc"] >= 18, f"F3 not refuted: escaped only {c_fail['esc']} at the cliff"
assert c_fail["stall"] > 0.50, f"cliff is not a contact-loss event: stall {c_fail['stall']}"
assert c_fail["launch"] < 0.10, "cliff looks like a launch (N213 channel), not a slide-off"
print(f"  -> the head loses contact ({c_fail['stall']:.2f} contact-free fraction), the lateral "
      f"servo keeps driving it, it slides off the edge and trips the 0.60 m workspace escape "
      f"test. launch_frac {c_fail['launch']:.4f} rules N213's launch channel OUT.")

# --- F4 (CONFIRMED): both modes, same channel, within one ladder step of their own reach ---
print()
print("=" * 78)
print("N215.5  F4 (CONFIRMED): raster and trochoid fail in the same band and through the same "
      "channel")
print("=" * 78)
print(f"trochoid cliff {cl_v} (step {cl_v[1] - cl_v[0]:.4f}) | raster cliff {cl_ras} "
      f"(step {cl_ras[1] - cl_ras[0]:.4f}) | separation {cl_v[1] - cl_ras[1]:.4f}")
assert cl_v[0] - cl_ras[1] <= 0.010, "F4 falsified: cliffs differ by more than one step"
ras_fail = agg(load("ras_fv030")[0], "fixture_B")
assert ras_fail["esc"] > 0 and ras_fail["stall"] > 0.50, "raster fails through another channel"
print(f"  raster at face_v 0.030: esc {ras_fail['esc']}/20, stall_frac {ras_fail['stall']:.3f} -> "
      f"same channel, and the 1-step offset is exactly the two plans' reaches")

# --- F5: the certificate is a region; the frozen face is NOT near its edge ---------------
print()
print("=" * 78)
print("N215.6  F5 (region CONFIRMED / 'sits close to its edge' REFUTED)")
print("=" * 78)
m_v, m_u = FACE_FROZEN["elong_v"] / cl_v[1], FACE_FROZEN["elong_u"] / cl_u[1]
m_r = FACE_FROZEN["round_r"] / cl_r[1]
print(f"frozen face margin at the measured cliff: v {FACE_FROZEN['elong_v']}/{cl_v[1]:.4f} = "
      f"{m_v:.2f}x | u {FACE_FROZEN['elong_u']}/{cl_u[1]:.4f} = {m_u:.2f}x | round "
      f"{FACE_FROZEN['round_r']}/{cl_r[1]:.4f} = {m_r:.2f}x")
print("pre-registered claim: v margin 1.27x is the TIGHTEST term in the segment")
assert m_v > 1.27, "the refutation of the 1.27x claim failed"
assert m_r > 1.27, "the round radius is also looser than the claimed tightest term"
assert m_r < m_u, "the tightest FACE term should be the round radius, not the across-patch v"
b_cert_v = [f for f in V_LADDER if f[1] >= cl_v[1]]
for name, face in b_cert_v:
    c = agg(cand(name), "fixture_B")
    assert c["cov"] == 1.0 and c["succ"] == 20 and c["esc"] == 0, f"certificate broken at {face}"
print(f"fixture_B certificate region: face_v in [{cl_v[1]:.4f}, {FACE_FROZEN['elong_v']}] "
      f"({FACE_FROZEN['elong_v'] / cl_v[1]:.2f}x) and face_u in [{cl_u[1]:.4f}, "
      f"{FACE_FROZEN['elong_u']}] ({FACE_FROZEN['elong_u'] / cl_u[1]:.2f}x), all at 1.0000 / 20-20")

# --- the round face degrades GRADED, the box edges are KNIFE ---------------------------
print()
print("=" * 78)
print("N215.7  cliff SHARPNESS is a face-SHAPE property (graded disc vs flat edge)")
print("=" * 78)
graded = [(f, agg(cand(n), "fixture_A")["cov"])
          for n, f in R_LADDER if f < FACE_FROZEN["round_r"]]
print("  round radius ->", " ".join(f"{f:.3f}:{c:.3f}" for f, c in graded))
mid = [c for f, c in graded if 0.5 < c < 0.999]
assert len(mid) >= 4, f"round cliff is not graded: only {len(mid)} intermediate steps"
box = [(f, agg(cand(n), "fixture_B")["cov"])
       for n, f in V_LADDER if f < FACE_FROZEN["elong_v"]]
print("  elongated v  ->", " ".join(f"{f:.4f}:{c:.3f}" for f, c in box))
assert sum(1 for _f, c in box if 0.5 < c < 0.999) == 0, "the box v cliff should be a step"
print("  -> a flat edge PARALLEL to the plan's excursion gives a knife edge; a convex boundary the "
      "plan's corner crosses obliquely gives a graded loss. Same channel, different geometry.")

# --- the face margin is spent by the SAME plan-frame error as N213 ---------------------
print()
print("=" * 78)
print("N215.8  the face requirement is ADDITIVE in the plan-frame offset (N213's term)")
print("=" * 78)
for name, noise in NOISE_ARMS:
    c = agg(cand(name), "fixture_B")
    print(f"  face_v {0.140 if name == 'id' else (0.060 if name.endswith('060n') else 0.055):.4f} "
          f"pose noise {noise}: cov {c['cov']:.4f} succ {c['succ']:2d}/20 esc {c['esc']:2d}")
noise60 = agg(load("tro_fv060n")[0], "fixture_B")
assert noise60["cov"] < 0.95, "the (0.03,6) band should lose coverage at face_v 0.060"
print("  -> the same head that certifies at the frozen 0.140 loses coverage at 0.060 under "
      "(0.03,6), i.e. face_half must cover plan_reach + the REALISED offset (N213: up to 3.63x "
      "the labelled sigma). The face law is not a constant: it is a budget shared with N213.")

print()
print("N215 ASSERTIONS PASS: identity bit-identical on the scored metric; keep=false in all "
      f"{len(keeps)} arms; F1 and F4 confirmed; F2, F3 and F5's second half refuted and left "
      "refuted; the cliff is the commanded plan's own reach.")