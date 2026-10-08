import json,glob,math,statistics as st,sys
def split(p):
    rows=[json.loads(l) for l in open(p)]
    eps=[r for r in rows if r.get('record')=='episode']
    modes=[r for r in rows if r.get('record')=='summary_mode']
    n=sum(s['n'] for s in modes[0]['per_suite'].values()); assert n*2==len(eps)
    return eps[:n],eps[n:],next((r for r in rows if r.get('record')=='compare'),None),rows
def q(v,p):
    v=sorted(v)
    if not v: return float('nan')
    k=(len(v)-1)*p; f=math.floor(k); c=math.ceil(k)
    return v[f] if f==c else v[f]+(v[c]-v[f])*(k-f)
def S(arm,suite,key='reg_err_xy_m',scale=1000.0):
    s=[r for r in arm if r['suite']==suite]
    v=[r[key]*scale for r in s if r.get(key) is not None and not math.isnan(r[key])]
    return dict(n=len(s),succ=sum(r['success'] for r in s),covc=st.mean(r['coverage_cont'] for r in s),
        p50=q(v,.5),p90=q(v,.9),mean=st.mean(v),casts=sorted({r.get('reg_casts') for r in s}),
        minmarg=min((r['reg_cn_margin'] for r in s if r.get('reg_cn_margin') is not None),default=None),
        wall=st.mean(r['wall_s'] for r in s),
        harn=sum(1 for r in s if r.get('backend')!='pybullet'))
def rays(k,cn,n): return k*k*cn*cn+n*n
if __name__=='__main__':
    print(f"{'file':34s} {'suite':9s} {'cSucc':>6s} {'bSucc':>6s} {'cCovc':>7s} {'bCovc':>7s} {'cP90mm':>7s} {'bP90mm':>7s} {'cP50':>6s} {'casts':>6s} {'minmarg':>7s} {'cWall':>6s} {'bWall':>6s} {'harn':>4s}")
    for f in sys.argv[1:]:
        c,b,cmp,rows=split(f); nm=f.split('/')[-1][:-6]
        hdr=next(r for r in rows if r.get('record')=='header')
        for suite in ('fixture_A','fixture_B','fixture_R'):
            a=S(c,suite); d=S(b,suite)
            cb=cmp.get(suite,{}) if cmp else {}
            print(f"{nm:34s} {suite:9s} {a['succ']:2d}/{a['n']:<3d} {d['succ']:2d}/{d['n']:<3d} {a['covc']:7.4f} {d['covc']:7.4f} {a['p90']:7.2f} {d['p90']:7.2f} {a['p50']:6.2f} {str(a['casts']):>6s} {str(a['minmarg']):>7s} {a['wall']:6.3f} {d['wall']:6.3f} {a['harn']+d['harn']:4d}", end='')
            if suite=='fixture_B' and cmp: print(f"  | welch={cb.get('p_value')} fisher={cb.get('fisher_p')} keep={cmp.get('keep')} | pose={hdr.get('pose_noise_cfg')} nCand={cmp['candidate_knobs'].get('REG_N')} nBase={cmp['baseline_knobs'].get('REG_N')} cn={cmp['candidate_knobs'].get('REG_CN')} df={cmp['candidate_knobs'].get('REG_DFACT')}")
            else: print()
