"""Local RTX: FastVLA smolvla 4-bit load + VRAM verdict. venv python."""
import json, time, torch
out = {}
def log(m): print(m, flush=True)
t0=time.time()
try:
    from fastvla import FastVLAModel
    m = FastVLAModel.from_pretrained("smolvla", load_in_4bit=True, use_peft=True)
    n = sum(p.numel() for p in m.parameters())
    out = {"params": n, "load_s": round(time.time()-t0,1),
           "alloc_mb": round(torch.cuda.memory_allocated()/1e6,1),
           "reserved_mb": round(torch.cuda.memory_reserved()/1e6,1)}
    log(f"LOADED: {out}")
except Exception as e:
    import traceback
    out = {"error": f"{type(e).__name__}: {str(e)[:250]}"}
    log(traceback.format_exc()[-700:])
print("FV_LOAD " + json.dumps(out)[:600], flush=True)
