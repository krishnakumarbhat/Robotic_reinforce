"""Compact archive of the auto-research session family before DB prune.
Keeps: title, dates, counts, models, cost/tokens, ALL user texts, assistant text
parts truncated to 250 chars, tool-call counts. Drops: reasoning, tool I/O,
snapshots, step markers.
Reads the LIVE db read-only; writes session_archive/MISSION_FAMILY_ARCHIVE.md.
"""
import datetime
import json
import os
import sqlite3
from collections import Counter

DB = "/home/pope/.local/share/opencode/opencode.db"
OUT = "session_archive/MISSION_FAMILY_ARCHIVE.md"
KEEP = {"ses_f486e6678ffeFOG4gzZ5RHJSpG"}


def ts(ms):
    try:
        return datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


con = sqlite3.connect(f"file://{DB}?mode=ro", uri=True)
fam = [r[0] for r in con.execute(
    "select id from session where title like '%MISSION%' or title like "
    "'%DP-Flow%' or title like '%Robotic%' or title like '%utoresearch%' or "
    "lower(title) like '%aegis%'")]
kids = [r[0] for r in con.execute(
    "select id from session where parent_id is not null and parent_id != '' "
    f"and parent_id in ({','.join('?' * len(fam))})", fam)]
targets = sorted(set(fam) | set(kids)) 
targets = [s for s in targets if s not in KEEP]
print(f"archiving {len(targets)} sessions", flush=True)

os.makedirs("session_archive", exist_ok=True)
meta = {r[0]: r[1:] for r in con.execute(
    "select id, title, agent, model, cost, tokens_input, tokens_output, "
    "time_created, time_updated from session")}
out = ["# Auto-research family archive (compact, pre-prune)",
       f"Sessions: {len(targets)}. Kept live: {sorted(KEEP)}.",
       "Per session: meta + all user texts + assistant text heads (250ch) + tool counts.",
       ""]
for i, sid in enumerate(targets):
    m = meta.get(sid, ("?", "?", "?", 0, 0, 0, 0, 0))
    title, agent, model, cost, ti, to, tc, tu = m
    nmsg = con.execute("select count(*) from message where session_id=?",
                       (sid,)).fetchone()[0]
    users, aheads, tools = [], [], Counter()
    for (mid, data) in con.execute(
            "select id, data from message where session_id=? order by time_created",
            (sid,)):
        try:
            d = json.loads(data)
        except Exception:
            continue
        role = d.get("role", "?")
        for (pd,) in con.execute(
                "select data from part where message_id=?", (mid,)):
            try:
                p = json.loads(pd)
            except Exception:
                continue
            t = p.get("type")
            if t == "text" and p.get("text", "").strip():
                if role == "user":
                    users.append(p["text"][:2000])
                else:
                    aheads.append(p["text"][:250].replace("\n", " "))
            elif t == "tool":
                tools[p.get("tool", "?")] += 1
    out.append(f"## {sid[:20]} | {title}")
    out.append(f"- created {ts(tc)} updated {ts(tu)} agent={agent} model={model}")
    out.append(f"- msgs={nmsg} cost={round(cost or 0, 2)} tok_in={ti} tok_out={to}")
    out.append(f"- tools={dict(tools.most_common(8))}")
    for u in users[:12]:
        out.append(f"- USER: {u[:600].replace(chr(10), ' ')}")
    if len(users) > 12:
        out.append(f"- ... +{len(users) - 12} more user msgs")
    for a in aheads[:15]:
        out.append(f"- SAY: {a}")
    if len(aheads) > 15:
        out.append(f"- ... +{len(aheads) - 15} more text parts")
    out.append("")
    if i % 200 == 0:
        print(f"  {i}/{len(targets)}", flush=True)

with open(OUT, "w") as f:
    f.write("\n".join(out))
print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.1f}MB")
