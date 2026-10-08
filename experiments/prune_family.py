"""Batched prune of the auto-research family (lock-friendly for live TUI).
100 sessions/batch with commit + incremental_vacuum each batch. Archive first
(archive_family.py) -- this only deletes. Run once, detached.
"""
import sqlite3

DB = "/home/pope/.local/share/opencode/opencode.db"
KEEP = {"ses_f486e6678ffeFOG4gzZ5RHJSpG"}
KEEP_TITLE = ("%cheat sheet%", "%Resume Customizer%")
BATCH = 100

con = sqlite3.connect(DB, timeout=300)
fam = {r[0] for r in con.execute(
    "select id from session where (title like '%MISSION%' or title like "
    "'%DP-Flow%' or title like '%Robotic%' or title like '%utoresearch%' or "
    "lower(title) like '%aegis%') and title not like ? and title not like ?",
    KEEP_TITLE)}
kids = {r[0] for r in con.execute(
    "select id from session where parent_id is not null and parent_id != '' "
    f"and parent_id in ({','.join('?' * len(fam))})", tuple(fam))}
targets = sorted((fam | kids) - KEEP)
print(f"deleting {len(targets)} sessions in batches of {BATCH}", flush=True)
TABLES = [("session_context_epoch", "session_id"),
          ("session_share", "session_id"),
          ("todo", "session_id"),
          ("part", "session_id"),
          ("message", "session_id"),
          ("event", "aggregate_id")]
done = 0
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
    print(f"  {done}/{len(targets)}", flush=True)
con.execute("PRAGMA incremental_vacuum")
con.commit()
print("prune done; run VACUUM manually when TUI idle for full reclaim",
      flush=True)
con.close()
