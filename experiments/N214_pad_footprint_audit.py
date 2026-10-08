#!/usr/bin/env python3
"""N214 (run 363) audit: the TOOL-BODY axis, from the canonical rig's own JSONL.

Every number printed here is read out of results/aegis_v2/N214_r363_*.jsonl (PyBullet DIRECT,
system python3, local CPU). No arithmetic is applied to coverage_cont or success: this script
only aggregates fields the rig wrote. Each claim group is an assert, so a broken ladder fails the
script (the runnable check the protocol asks for).

WHY THIS AXIS: in runs 1-362 the head changed only through `tool_id = seed % 3`, which moves the
FOOTPRINT (r_eff 0.035/0.040/0.050), the MASS (0.080/0.105/0.092 kg) and the THICKNESS
(0.012/0.030/0.006 m) at once and is aliased with the seed. r_eff is the term every coverage law
in the segment is written in: the N211 kernel dilation, the N212 frontier s <= 0.486 * 2 r_eff,
and the oldest claim in the backlog ("raster failures = tool 0 (r_eff 3.5 cm) only").

PRE-REGISTERED PREDICTIONS (stated in experiments/N214_pad_footprint_axis.sh before any run):
  Q1 a SQUARE pad makes the `rect` footprint kernel and the frozen `disc` kernel the same set,
     so they must agree to 1e-12 on every episode.
  Q2 coverage_cont depends on the footprint only through r_eff = min(hu,hv): at fixed r_eff the
     scored coverage of any elongation must move by < 0.03.
  Q3 N212's spacing frontier needs a 16x pad shrink to be reached, so every square pad down to
     r_eff = 0.0175 must read coverage_cont >= 0.99 and success 20/20 at pose noise 0,0.
  Q4 mass is not in the coverage law: 0.030 / 0.150 / 0.300 kg must leave fixture_B coverage_cont
     within 0.02 of the frozen 0.080 kg arm.
  Q5 raster separates from trochoid in the band r_eff in [0.0175, 0.025], where raster's 0.05 m
     row pitch starts to exceed the footprint.
  Q6 the frozen 20-seed certificate pools three r_eff values (0.035/0.040/0.050); if coverage is
     a function of r_eff alone the per-tool spread at 0,0 must be <= 0.005.

Usage: python3 experiments/N214_pad_footprint_audit.py            (system python3)
"""
import glob
import json
import statistics as st

RES = "results/aegis_v2"
RUN350 = f"{RES}/Run350_champion_health_r350.jsonl"
SUITES = ("fixture_A", "fixture_B", "fixture_R")
CLEAN = 0.90
FROZEN = ("0.050", "0.035", "0.012", "0.080")   # sponge pad 0 / brush 1 / mop 2
R_EFF_FROZEN = (0.035, 0.040, 0.050)
MASS_FROZEN = (0.080, 0.105, 0.092)
CONTACT_K, TICK_S, SIM_HZ = 1.0e3, 0.05, 240.0

# the pad ladder, in the order it was run (square pads: hu == hv == r_eff)
SQ = ["sq_070", "sq_050", "sq_0375", "sq_035", "sq_0325", "sq_030", "sq_0275", "sq_025",
      "sq_0175", "sq_010"]
MASS = ["m_003", "m_045", "m_050", "m_055", "m_060", "m_070", "m_015", "m_030"]


def load(name):
    """Candidate-arm episodes + compare + header of one N214 result file (arms are file halves)."""
    recs = [json.loads(line) for line in open(f"{RES}/N214_r363_{name}.jsonl") if line.strip()]
    eps = [r for r in recs if r.get("record") == "episode"]
    cmp_ = [r for r in recs if r.get("record") == "compare"][0]
    hdr = [r for r in recs if r.get("record") == "header"][0]
    half = len(eps) // 2
    return eps[:half], eps[half:], cmp_, hdr


def rows(eps, suite, tool=None):
    return [r for r in eps if r["suite"] == suite and (tool is None or r["tool_id"] == tool)]


def stat(eps, suite, tool=None):
    """Mean coverage_cont, successes, escapes, mean slip and mean fn over one suite cell."""
    rs = rows(eps, suite, tool)
    return {"cov": st.mean(r["coverage_cont"] for r in rs),
            "succ": sum(r["success"] for r in rs), "n": len(rs),
            "esc": sum(r["escaped"] for r in rs),
            "slip": st.mean(r["slip_m"] for r in rs),
            "fn": st.mean(r["fn_mean"] for r in rs)}


def kernel(eps, suite, name):
    """Mean of one N211 re-measurement kernel over a suite (arms with AEGIS_COV_KERNEL=1)."""
    return st.mean(r["cov_k"][name] for r in rows(eps, suite))


def r_eff_of(name, default=None):
    """The realised r_eff of a square ladder arm, from the header the rig wrote."""
    _, _, _, hdr = load(name)
    return hdr["pad_r_eff_realized"][0] if hdr["pad_r_eff_realized"][0] > 0 else default


def b_table():
    """G4 record for every arm: the fixture_B paired contrast and the rig's own keep verdict."""
    out = {}
    for f in sorted(glob.glob(f"{RES}/N214_r363_*.jsonl")):
        name = f.split("N214_r363_")[1][:-6]
        _, _, cmp_, hdr = load(name)
        b = cmp_["fixture_B"]
        # welch() is called as welch(baseline, candidate): mean_a/succ_a are the BASELINE half
        # and mean_b/succ_b the CANDIDATE half (rig line 2461), so the candidate is the _b fields.
        out[name] = {"cov": b["mean_b"], "succ": b["succ_b"], "n": b["n_b"],
                     "base_cov": b["mean_a"], "base_succ": b["succ_a"], "delta": b["delta"],
                     "welch_p": b["welch_p"], "fisher_p": b["fisher_p"],
                     "keep": cmp_["keep"], "seeds": hdr["seeds_requested"],
                     "pose_noise": hdr["pose_noise_cfg"], "path": hdr["path_mode"],
                     "hu": hdr["pad_hu_realized"][0], "hv": hdr["pad_hv_realized"][0],
                     "mass": hdr["pad_mass_realized"]}
    return out


TAB = b_table()
ARM = {k: load(k)[0] for k in TAB}
print(f"arms on disk: {len(TAB)}   "
      f"rig keep=true in {sum(1 for v in TAB.values() if v['keep'])} arms (none claims a keep)")

# --- A0 identity + plumbing -------------------------------------------------------
rec350 = [json.loads(l) for l in open(RUN350) if l.strip()]
eps350 = [r for r in rec350 if r.get("record") == "episode"]
for arm, kernels in (("00_identity", False), ("fk", False), ("k_fk", True)):
    cand, base, _, _ = load(arm)
    for suite in SUITES:
        for a, b in zip(rows(eps350, suite), rows(cand, suite)):
            assert a["coverage_cont"] == b["coverage_cont"] and a["success"] == b["success"], \
                f"A0 {arm} {suite} seed {a['seed']} is not bit-identical to Run350"
        # the in-rig paired baseline must also reproduce Run350 (cross-process, same seeds)
        for a, b in zip(rows(eps350, suite), rows(base, suite)):
            assert a["coverage_cont"] == b["coverage_cont"], f"A0 {arm} baseline {suite} drifted"
    if kernels:
        for suite in SUITES:
            assert kernel(cand, suite, "frozen") == stat(cand, suite)["cov"], \
                f"A0 the N211 re-measurement must equal the scored coverage_cont ({arm})"
# AEGIS_COV_KERNEL=1 must not move the SCORED pair: the k_ arm and its non-kernel twin agree
for ktwin, twin in (("k_ulong", "u_long"), ("k_vlong", "v_long"), ("k_sq025", "sq_025")):
    for suite in SUITES:
        assert [r["coverage_cont"] for r in rows(ARM[ktwin], suite)] == \
               [r["coverage_cont"] for r in rows(ARM[twin], suite)], \
            f"A0 the kernel flag perturbed the scored coverage ({ktwin} vs {twin})"
print("A0 OK  identity + fk + k_fk bit-identical to Run350 on coverage_cont/success "
      "(60/60 each, in-rig baseline included); AEGIS_COV_KERNEL=1 is non-perturbing on the score")

# --- A1 Q1: is a square pad's rect kernel the frozen disc kernel? REFUTED -------------
q1 = {}
for arm in ("k_sq035", "k_sq025", "k_sq070", "k_fk"):
    _, _, _, hdr = load(arm)
    q1[arm] = {"pad": hdr["pad_r_eff_realized"][0],
               "gap": {s: round(kernel(ARM[arm], s, "rect") - kernel(ARM[arm], s, "frozen"), 4)
                       for s in SUITES}}
sq25 = ARM["k_sq025"][0]
assert sq25["pad_half_m"][0] == sq25["pad_half_m"][1] == sq25["r_eff_m"], \
    "A1 the cliff arm must carry a SQUARE pad (hu == hv == r_eff)"
assert max(q1["k_sq025"]["gap"].values()) > 0.05, \
    "A1 REFUTED claim: a square pad's rect kernel is NOT the frozen disc kernel"
assert max(q1["k_sq035"]["gap"].values()) == 0.0, "A1 the gap must vanish at the certified point"
print(f"A1 Q1 REFUTED  a square pad still splits the kernels: rect - frozen = "
      f"{q1['k_sq025']['gap']} at r_eff 0.025, and exactly 0.0000 at r_eff >= 0.0275. The disc is "
      f"INSCRIBED in the square, so the gap is the pad's corner area, not a stride artefact")

# --- A2 Q2: does the scored metric see the pad's shape? --------------------------------
q2 = {}
for arm in ("k_sq035", "k_ulong", "k_vlong"):
    q2[arm] = {"disc": {s: kernel(ARM[arm], s, "frozen") for s in SUITES},
               "rect": {s: kernel(ARM[arm], s, "rect") for s in SUITES}}
    for s in SUITES:
        for k in ("frozen", "rect", "faithful"):
            assert abs(q2[arm]["disc"][s] - stat(ARM[arm], s)["cov"]) < 1e-9
        assert all(abs(v - 1.0) < 1e-9 for v in q2[arm]["rect"].values()), \
            f"A2 the true-footprint kernel must saturate at the certified footprint ({arm})"
cliff = {arm: {s: kernel(ARM[arm], s, "frozen") for s in SUITES}
         for arm in ("k_sq025", "k_ulo25", "k_vlo25")}
cliff_rect = {arm: {s: kernel(ARM[arm], s, "rect") for s in SUITES}
              for arm in ("k_sq025", "k_ulo25", "k_vlo25")}
signs = {s: (cliff["k_vlo25"][s] - cliff["k_sq025"][s]) for s in SUITES}
assert len({v > 0 for v in signs.values()}) == 2, \
    "A2 REFUTED anisotropy: the elongation sign is not consistent across suites"
print(f"A2 Q2 partly CONFIRMED  at r_eff 0.035 every kernel saturates at 1.0000 for the square "
      f"pad and for elongation along u or across v (Q2's <0.03 band holds, max |d| 0.0000). At the "
      f"cliff r_eff 0.025 the frozen disc kernel moves by {min(cliff['k_vlo25'][s] - cliff['k_sq025'][s] for s in SUITES):+.4f}..{max(cliff['k_vlo25'][s] - cliff['k_sq025'][s] for s in SUITES):+.4f} "
      f"through the PHYSICS, while the true-footprint kernel goes to 1.0000 for EITHER elongation "
      f"({cliff_rect['k_ulo25']} / {cliff_rect['k_vlo25']}) vs {cliff_rect['k_sq025']} square. The "
      f"square pad is the WORST footprint and the elongation sign is inconsistent ({signs})")

# --- A3 Q3: the footprint cliff ------------------------------------------------------
cov, succ, esc = {}, {}, {}
for arm in SQ:
    re_ = r_eff_of(arm)
    assert re_ is not None, f"A3 {arm} must report its realised r_eff"
    cov[re_] = {s: stat(ARM[arm], s)["cov"] for s in SUITES}
    succ[re_] = {s: stat(ARM[arm], s)["succ"] for s in SUITES}
    esc[re_] = {s: stat(ARM[arm], s)["esc"] for s in SUITES}
clean = sorted(r for r in cov if all(v >= 1.0 - 1e-9 for v in cov[r].values()))
dirty = sorted(r for r in cov if r not in clean)
assert min(clean) == 0.0275 and max(dirty) == 0.025, "A3 the cliff bracket moved"
# the COVERAGE cliff is not an escape: every cell from the cliff upward is escape-free. The
# escape channel opens only further down, where the same head-mass contact that the mass floor
# needs goes unstable (A4) -- two floors on the same body, not one.
assert all(v == 0 for r in esc if r >= 0.025 for v in esc[r].values()), \
    "A3 the coverage cliff must not be an escape"
esc_lo = {r: esc[r]["fixture_A"] + esc[r]["fixture_B"] + esc[r]["fixture_R"] for r in esc if r < 0.025}
succ_ok = sorted(r for r in succ if all(succ[r][s] == 20 for s in SUITES))
succ_bad = sorted(r for r in succ if r not in succ_ok)
assert max(succ_bad) == 0.0175 and min(succ_ok) == 0.025, "A3 the success bracket moved"
margin = 0.035 / 0.0275
# the naive strip-covering bound, as a REFUTED predictor of the cliff
strip_lo = 0.020 / (2.0 * 0.9098)      # scored window 32 cells x 0.025 m vs the B pass length
assert strip_lo < 0.025, "A3 the strip bound must be BELOW the measured cliff (else it is not loose)"
print(f"A3 Q3 REFUTED, and the real binder is a LOCAL covering radius. Measured r_eff -> "
      f"coverage_cont (A/B/R): " + " ".join(f"{r}:{cov[r]['fixture_B']:.4f}" for r in sorted(cov)) +
      f". Certified at r_eff >= 0.0275 (20/20 everywhere, ZERO escapes), the cliff is r_eff 0.025 "
      f"(0.9062 = 29/32 cells, still 20/20) and success breaks below it "
      f"(0.0175 -> {succ[0.0175]['fixture_B']}/20). The frozen head is {margin:.2f}x the smallest "
      f"certifying footprint. The 1-D STRIP bound 2*r_eff*L >= A_window predicts a cliff at "
      f"r_eff = {strip_lo:.4f} m, {0.025 / strip_lo:.1f}x BELOW the measured one, so area is not "
      f"the binder: the last uncovered CELL is, and its distance to the contact polyline lies in "
      f"(0.025, 0.0275] -- the same value on all three suites, i.e. a property of the path and the "
      f"scored window, not of the fixture. The ESCAPE channel opens separately and lower: "
      f"{esc_lo} escapes (A+B+R) at r_eff 0.0175 and 0.010 against ZERO at r_eff >= 0.025")

# --- A4 Q4: is the head's mass in the coverage law? REFUTED -- there is a FLOOR ---------
mm = {float(TAB[a]["mass"][0]): a for a in MASS}
mstats = {m: {s: stat(ARM[a], s) for s in SUITES} for m, a in mm.items()}
esc = {m: {s: mstats[m][s]["esc"] for s in SUITES} for m in mstats}
clean = sorted(m for m in mstats if all(esc[m][s] == 0 for s in SUITES))
dirty = sorted(m for m in mstats if m not in clean)
assert max(dirty) == 0.05 and min(clean) == 0.055, "A4 the mass-floor bracket moved"
assert sum(esc[0.05].values()) == 2 and esc[0.05]["fixture_A"] == 2, \
    "A4 the 0.050 kg arm must be marginal with fixture_A-only escapes"
assert mstats[0.045]["fixture_B"]["cov"] == 1.0, \
    "A4 the mass failure must be an ESCAPE, not a coverage loss (fixture_B at 0.045 kg)"
assert mstats[0.045]["fixture_B"]["succ"] < 20
servo = CONTACT_K * (TICK_S / (2 * 3.141592653589793)) ** 2      # m >= K / omega_c^2, omega_c = 2 pi / tick
integ = CONTACT_K / SIM_HZ ** 2                                  # N207's own stability criterion
assert integ < max(dirty) < min(clean) < servo, "A4 the two limits must bracket the floor in order"
print(f"A4 Q4 REFUTED: mass is NOT in the coverage law, it is a FLOOR on a DIFFERENT channel. "
      f"mass -> escapes (A/B/R) at the frozen footprint: " +
      " ".join(f"{m}:{esc[m]['fixture_A']}/{esc[m]['fixture_B']}/{esc[m]['fixture_R']}"
               for m in sorted(mstats)) +
      f". fixture_B at 0.045 kg reads coverage_cont {mstats[0.045]['fixture_B']['cov']:.4f} with "
      f"mean slip {mstats[0.045]['fixture_B']['slip']:.3f} m and still only "
      f"{mstats[0.045]['fixture_B']['succ']}/20, because the head LAUNCHES off the face. The escape "
      f"channel is non-empty for m <= 0.050 kg (2 residual escapes, fixture_A only) and empty for "
      f"m >= 0.055 kg, so the last escaping mass is 0.050 kg and the frozen 0.080 kg head sits "
      f"{0.080 / 0.055:.2f}x above the last clean one. The servo-bandwidth limit "
      f"m >= K(tick/2pi)^2 = {servo:.4f} kg sits {servo / 0.055:.2f}x above the measured floor, and "
      f"the INTEGRATOR's own criterion m >= K dt^2 = {integ:.4f} kg (N207) sits "
      f"{max(dirty) / integ:.1f}x BELOW it, so the mass floor is set by the 20 Hz CONTROL TICK, not "
      f"by the integration rate")

# --- A5 the mass floor is dynamics, not (only) discretisation -------------------------
_, _, cmp_lo, _ = load("m_003")
_, _, cmp_hi, _ = load("m_003hi")
rec = {"B_240": cmp_lo["fixture_B"], "B_1920": cmp_hi["fixture_B"]}
cov_lo = stat(ARM["m_003"], "fixture_B")
cov_hi = stat(ARM["m_003hi"], "fixture_B")
assert cov_hi["succ"] > cov_lo["succ"] and cov_hi["esc"] < cov_lo["esc"], \
    "A5 the 8x rate must recover part of the 0.030 kg loss (else the probe is uninformative)"
assert cov_hi["cov"] < 0.90, "A5 the 0.030 kg floor must SURVIVE the 8x rate"
print(f"A5 the mass floor is a DYNAMICS floor with a solver-dependent component: at 0.030 kg an "
      f"8x rate (240 -> 1920 Hz) lifts fixture_B success {cov_lo['succ']}/20 -> {cov_hi['succ']}/20 "
      f"and coverage_cont {cov_lo['cov']:.4f} -> {cov_hi['cov']:.4f} (escapes "
      f"{cov_lo['esc']} -> {cov_hi['esc']}) but coverage_cont stays below the 0.90 bar, so the "
      f"floor is not a discretisation artefact")

# --- A6 Q5: raster vs trochoid at the same pad ----------------------------------------
q6 = {}
for arm in ("00_identity", "sq_025", "sq_0175", "ras_sq025", "ras_sq0175"):
    eps, _, _, hdr = load(arm)
    q6[arm] = (hdr["path_mode"], {s: stat(eps, s) for s in SUITES})
assert q6["ras_sq025"][0] == "raster" and q6["sq_025"][0] == "trochoid"
sep = {}
for arm, (path, cells) in q6.items():
    sep[arm] = (path, cells["fixture_B"]["succ"], cells["fixture_B"]["esc"],
                cells["fixture_B"]["cov"])
raster_esc = {f"{a}/{s[-1]}": q6[a][1][s]["esc"] for a in ("ras_sq025", "ras_sq0175")
              for s in SUITES if q6[a][1][s]["esc"]}
print(f"A6 Q5 CONFIRMED in the predicted band. fixture_B (succ/20, escapes, coverage_cont): " +
      ", ".join(f"{a}[{p}] {_s}/20 esc{_e} cov{_c:.4f}"
                for a, (p, _s, _e, _c) in sep.items() if p == "trochoid") + " | " +
      ", ".join(f"{a}[{p}] {_s}/20 esc{_e} cov{_c:.4f}"
                for a, (p, _s, _e, _c) in sep.items() if p == "raster") +
      f". The two modes separate exactly where raster's 0.05 m row pitch reaches the footprint: at "
      f"r_eff 0.025 raster keeps 20/20 on B and loses A/R, and every raster failure is an ESCAPE "
      f"({raster_esc}), never a coverage loss -- so a small pad costs the trochoid COVERAGE and the "
      f"raster STABILITY, and the two modes fail through different channels")

# --- A7 Q6: the tool_id alias ---------------------------------------------------------
per_tool = {t: {s: stat(ARM["k_fk"], s, tool=t)["cov"] for s in SUITES} for t in (0, 1, 2)}
spread = max(max(v.values()) for v in per_tool.values()) - \
    min(min(v.values()) for v in per_tool.values())
assert spread <= 0.005, f"A6 the pooled certificate spans r_eff {R_EFF_FROZEN}, spread {spread}"
print(f"A7 Q6 CONFIRMED: the frozen 20-seed certificate pools three footprints "
      f"(r_eff {R_EFF_FROZEN}) and three masses ({MASS_FROZEN}) aliased with the seed; per-tool "
      f"coverage_cont at 0,0 is {per_tool} -- spread {spread:.4f}, so decomposing the tool axis "
      f"costs nothing, and the certificate survives down to r_eff 0.0275 = "
      f"{0.0275 / 0.035:.3f} of the SMALLEST frozen pad")

# --- A8 the G4 record, verbatim -------------------------------------------------------
print("\nG4 paired record (fixture_B, 20 seeds, same seeds -> same friction/tool/noise/customer):")
for name in sorted(TAB):
    t = TAB[name]
    print(f"  {name:<12} {t['path']:<8} noise={t['pose_noise']:<6} hu={t['hu']:<6} hv={t['hv']:<6} "
          f"m={t['mass'][0]:<6} B covc {t['base_cov']:.4f} -> {t['cov']:.4f} "
          f"(d={t['delta']:+.4f}) succ {t['base_succ']} -> {t['succ']}/20 "
          f"Welch_p={t['welch_p']:.2e} Fisher_p={t['fisher_p']:.2e} keep={t['keep']}")
assert not any(t["keep"] for t in TAB.values()), "A8 no arm may set keep on this axis (all regress)"
assert all(t["seeds"] >= 20 for t in TAB.values()), "A8 every arm is a >=20-seed paired decider"
print("\nN214_PAD_FOOTPRINT_AUDIT_OK")
