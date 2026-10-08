"""AEGIS Colab bootstrap: subprocess-isolated launcher (run via `colab exec -f`).

Why this exists: Colab kernels preload torch (2.11) before any pip runs, so
in-cell installs of pinned torch NEVER take effect in the cell process
(bursts v2-v5 died on CUSTOM_KEY / skew). This cell never imports torch
itself: it pip-installs the pinned stack, then runs the training script as a
SUBPROCESS under a fresh interpreter, which resolves everything from disk.
(No venv: `python3 -m venv` is unavailable on Colab images, and a subprocess
already guarantees a clean module table.)

Usage (local, chained with `;` so stop always runs):
  colab upload -s S /path/colab_aegis_ft.py /content/aegis_run.py
  colab exec -s S -f /path/colab_aegis_boot.py --timeout 18000
"""
import subprocess
import sys

RUN_SCRIPT = "/content/aegis_run.py"
# Expert-only FT until the VM-only LoRA recursion is root-caused (locally
# unreproducible with identical pins). Flip to "0" to restore LoRA+expert.
os.environ.setdefault("AEGIS_NO_LORA", "1")
# Pinned to the EXACT locally-verified stack (venv, 2026-09-27). Floating peft
# (>=0.17) pulled a version whose LoRA layer recurses at forward; torch 2.11
# dropped CUSTOM_KEY; transformers 5.x breaks lerobot 0.4.4. Pin everything.
PINS = ["torch==2.10.0", "lerobot[smolvla,dataset]==0.4.4", "peft==0.21.0",
        "transformers==4.57.6", "huggingface_hub==0.35.3", "safetensors==0.8.0",
        "num2words"]


def sh(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


r = sh([sys.executable, "-m", "pip", "-q", "install"] + PINS,
       capture_output=True, text=True, timeout=2400)
print("boot pip rc=", r.returncode, flush=True)
if r.returncode != 0:
    print((r.stdout + r.stderr)[-1500:])
    raise SystemExit("boot pip failed")
# Colab image ships torchao built for torch 2.11: it hard-crashes under the
# pinned torch 2.10 (torchao.quantization -> torch._inductor chain). Nothing in
# our stack (lerobot/peft/transformers/SmolVLA/LoRA/flow) needs torchao.
r = sh([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"],
       capture_output=True, text=True, timeout=300)
print("boot un-torchao rc=", r.returncode, flush=True)
print("bootstrap OK: launching training in fresh subprocess", flush=True)
proc = subprocess.Popen([sys.executable, "-u", RUN_SCRIPT], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
taillines: list = []
assert proc.stdout is not None
for line in proc.stdout:
    taillines.append(line)
    if len(taillines) > 4000:
        del taillines[:1000]
    print(line, end="", flush=True)
rc = proc.wait()
open("/tmp/aegis_train.log", "w").write("".join(taillines))
print(f"training exit rc={rc}", flush=True)
print("--- child tail ---", flush=True)
for line in taillines[-15:]:
    print(line[:200], end="" if line.endswith("\n") else "\n", flush=True)
raise SystemExit(rc)
