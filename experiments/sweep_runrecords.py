"""Archive + delete in-project dead run-records and /tmp/model-probe orphans.
Keeps: live session + anything updated in the last 24h. Appends compact sections
to session_archive/MISSION_FAMILY_ARCHIVE.md, then batch-deletes + checkpoints.
Run once, detached.
"""
import datetime
import json
import os
import sqlite3
import time
from collections import Counter

DB = "/home/pope/.local/share/opencode/opencode.db"
OUT = "session_archive/MISSION_FAMILY_ARCHIVE.md"
KEEP = {"ses_f486e6678ffeFOG4gzZ5RHJSpG"}
DIRS = ("/media/pope/projecteo/github_proj/a_resume/Robotic_reinforce",
        "/tmp/model-probe")
CUTOFF = int(time.time() * 1000) - 24 * 3600 * 1000
BATCH = 200


def ts(ms):
    try:
        return datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


con = sqlite3.connect(DB, timeout=300)
cands = [r[0] for r in con.execute(
    f"select id from session where directory in ({','.join('?' * len(DIRS))}) "
    "and time_updated < ?", (*DIRS, CUTOFF))]
targets = sorted(set(cands) - KEEP)
print(f"archiving {len(targets)} run-records", flush=True)
meta = {r[0]: r[1:] for r in con.execute(
    "select id, title, agent, model, cost, tokens_input, tokens_output, "
    "time_created, time_updated from session")}
lines = ["", "# Run-record sweep (probes/workers, in-project + /tmp/model-probe)",
         f"Sessions: {len(targets)}. Keep rule: live + updated<24h.",
         ""]
for i, sid in enumerate(targets):
    m = meta.get(sid, ("?", "?", "?", 0, 0, 0, 0, 0))
    title, agent, model, cost, ti, to, tc, tu = m
    if isinstance(model, str) and model.startswith("{"):
        try:
            md = json.loads(model)
            model = md.get("modelID", model[:40])
        except Exception:
            pass
    nmsg = con.execute("select count(*) from message where session_id=?",
                       (sid,)).fetchone()[0]
    users, tools = [], Counter()
    for (mid, data) in con.execute(
            "select id, data from message where session_id=? order by time_created",
            (sid,)):
        try:
            d = json.loads(data)
        except Exception:
            continue
        for (pd,) in con.execute(
                "select data from part where message_id=?", (mid,)):
            try:
                p = json.loads(pd)
            except Exception:
                continue
            if p.get("type") == "text" and p.get("text", "").strip():
                if d.get("role") == "user":
                    users.append(p["text"][:500].replace("\n", " "))
            elif p.get("type") == "tool":
                tools[p.get("tool", "?")] += 1
    lines.append(f"## {sid[:20]} | {title}")
    lines.append(f"- {ts(tc)}->{ts(tu)} agent={agent} model={model} msgs={nmsg} tools={dict(tools.most_common(5))}")
    for u in users[:3]:
        lines.append(f"- USER: {u[:300]}")
    lines.append("")
    if i % 500 == 0:
        print(f"  arch {i}/{len(targets)}", flush=True)
with open(OUT, "a") as f:
    f.write("\n".join(lines))
print("archive appended", flush=True)

done = 0
TABLES = [("session_context_epoch", "session_id"),
          ("session_share", "session_id"),
          ("todo", "session_id"),
          ("part", "session_id"),
          ("message", "session_id"),
          ("event", "aggregate_id")]
for i in range(0, len(targets), BATCH):
    b = targets[i:i + BATCH]
    ph = ",".join("?" * len(b))
    for tbl, col in TABLES:
        try:
            con.execute(f"delete from {tbl} where {col} in ({ph})", b)
        except Exception as e:
            print(f"batch {i}: {tbl} ERR {e}", flush=True)
    con.execute(f"delete from session where id in ({ph})", b)
    con.commit()
    try:
        con.execute("PRAGMA incremental_vacuum(2000)")
    except Exception as e:
        print(f"batch {i}: vacuum ERR {e}", flush=True)
    done += len(b)
    print(f"  del {done}/{len(targets)}", flush=True)
con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
con.commit()
con.close()
print("sweep done", flush=True)
