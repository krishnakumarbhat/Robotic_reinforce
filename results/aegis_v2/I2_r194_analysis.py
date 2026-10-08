"""T194 per-tool conformal admission, POST-HOC ONLY (director iter 19). Pre-reg, single eval.

Rule (train-locked on CALIB, frozen R173 files, zero rig edits):
  Admit(e) := T173(e) AND jerk(e) <= THETA[tool_id(e)]
  THETA[t] = 90th percentile of jerk over CALIB SUCCESS episodes with tool t (alpha=0.1).
  T173(e) := jerk<=0.618 AND jerk<=0.014133 AND stall_frac<=0.05 (bit-identity asserted).
Held-out eval: pooled calib? No -- held-out = R173 TEST files (BASE_SEED=190000).
Metrics: P(fail|admit) vs T173 diode, recall/score_sel, coverage, archive adm frac.
"""
import json, statistics as st

THETA_FROZEN = 0.014133
CAL = 'results/aegis_v2/I2_r173_calib_0012.jsonl'
TST = 'results/aegis_v2/I2_r173_test_0012.jsonl'
ARC = 'results/aegis_v2/I2_r173_archive_20.jsonl'

def eps(f):
    return [json.loads(l) for l in open(f) if json.loads(l).get('record') == 'episode']

def t173(e):
    return e['jerk'] <= 0.618 and e['jerk'] <= THETA_FROZEN and e['stall_frac'] <= 0.05

def pct(q, xs):
    xs = sorted(xs); i = min(len(xs) - 1, max(0, int(q * len(xs))))
    return xs[i]

cal, tst, arc = eps(CAL), eps(TST), eps(ARC)
# train-locked thresholds from CALIB SUCCESS per tool
THETA, NB = {}, {}
for t in (0, 1, 2):
    js = [e['jerk'] for e in cal if e['tool_id'] == t and e['success']]
    NB[t] = len(js); THETA[t] = pct(0.9, js)
assert all(n >= 20 for n in NB.values()), f'sparse tool bin, merge per spec: {NB}'

def admit(e):
    return t173(e) and e['jerk'] <= THETA[e['tool_id']]

def report(rows):
    adm = [e for e in rows if admit(e)]
    base = [e for e in rows if t173(e)]
    def pf(rs):
        return sum(1 for e in rs if not e['success']) / len(rs) if rs else 0.0
    fails = [e for e in rows if not e['success']]
    rec = sum(1 for e in fails if not admit(e)) / len(fails) if fails else 1.0
    cov = st.mean(e['coverage'] for e in adm) if adm else 0.0
    cov0 = st.mean(e['coverage'] for e in base) if base else 0.0
    return {'n': len(rows), 'n_adm': len(adm), 'p_fail_adm': pf(adm),
            'p_fail_t173': pf(base), 'recall_sel': rec, 'score_sel': 100 * rec,
            'cov_adm': cov, 'cov_t173': cov0}

out = {'THETA_tool': THETA, 'N_calib_succ_tool': NB,
       'theta_frozen_bitident': THETA_FROZEN,
       'held': report(tst), 'calib': report(cal), 'archive_adm_frac':
       sum(1 for e in arc if admit(e)) / len(arc)}
# bit-identity: T173 conjuncts inert?
out['i7_binds_held'] = sum(1 for e in tst if e['jerk'] > 0.618)
out['stall_binds_held'] = sum(1 for e in tst if e['stall_frac'] > 0.05)
h = out['held']
out['lift_vs_t173_pts'] = 100 * (h['p_fail_t173'] - h['p_fail_adm'])
json.dump(out, open('results/aegis_v2/I2_r194_result.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
