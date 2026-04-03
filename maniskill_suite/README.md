# ManiSkill Low-VRAM Experiment Suite

This folder is a clean ManiSkill research scaffold aimed at Linux plus NVIDIA hardware with about 4 GB of VRAM. It does not depend on the deleted `panda-gym` working tree and can be used as a separate experiment layer inside this repository.

What is included:

- a 21-combination experiment matrix across `PickCube-v1`, `PushCube-v1`, and `StackCube-v1`
- per-combination report folders under `experiments/`
- setup and data scripts for ManiSkill install, teleoperation, demo replay, and official demo download
- a command generator so every experiment can be launched from one matrix file
- research notes and experiment tables written in Markdown

What is intentionally not claimed:

- no baseline has been executed yet in this repository
- no result numbers are fabricated in the reports
- Vulkan and NVIDIA driver installation are documented but not automated with `sudo`

## Why This Layout

Your research note is centered on imitation learning plus reinforcement learning inside a Real-Sim-Real style workflow. This suite mirrors that logic with a low-compute bias:

- teleop or downloaded demos provide the prior
- BC, ACT, Diffusion Policy, RFCL, and RLPD cover learning from demos
- PPO and SAC provide RL-only baselines
- fair evaluation is based on `success_once`, `success_at_end`, and return

## Folder Layout

```text
maniskill_suite/
├── README.md
├── requirements.txt
├── metrics_template.json
├── experiment_matrix.json
├── data/
│   └── README.md
├── scripts/
│   ├── bootstrap_maniskill.sh
│   ├── collect_teleop_demos.sh
│   ├── download_official_demos.sh
│   ├── replay_demos.sh
│   ├── run_from_matrix.py
│   ├── print_experiment_plan.py
│   └── summarize_results.py
├── reports/
│   ├── 00_overview.md
│   ├── 01_low_vram_strategy.md
│   ├── 02_algorithm_mapping.md
│   ├── 03_experiment_table.md
│   ├── 04_mouse_cursor_il_workflow.md
│   ├── 05_breakthrough_hypotheses.md
│   ├── 06_runtime_notes.md
│   └── experiment_results.md
└── experiments/
    └── <one folder per task/algorithm combination>
```

## Quick Start

1. Create a Python environment and install the core packages.

```bash
cd maniskill_suite
bash scripts/bootstrap_maniskill.sh
```

2. Verify the core install with a no-render smoke test.

```bash
./.venv/bin/python -m mani_skill.examples.demo_random_action -e PickCube-v1 --render-mode none -n 1
```

3. Collect a small teleop pilot with mouse plus keyboard.

```bash
bash scripts/collect_teleop_demos.sh PickCube-v1
```

4. Replay the compressed demos into a state dataset for IL baselines.

```bash
bash scripts/replay_demos.sh PickCube-v1 teleop state pd_ee_delta_pos 3 physx_cpu
```

5. Print the full experiment plan.

```bash
./.venv/bin/python scripts/print_experiment_plan.py
```

6. Print the command for a specific combination.

```bash
./.venv/bin/python scripts/run_from_matrix.py --combo pickcube_bc
```

7. After you run experiments, place `metrics.json` inside each combination folder and summarize them.

```bash
./.venv/bin/python scripts/summarize_results.py
```

## 4 GB VRAM Rules

- Start with `obs_mode=state` for every first-pass run.
- Keep `render_mode` off during training.
- Use `physx_cpu` for trajectory replay unless you prove the GPU backend is stable.
- Use official RGB baselines only after a state-based run is working.
- Expect RFCL and RLPD to need extra JAX dependencies not installed by default.

## Teleoperation Notes

ManiSkill's click-and-drag teleoperation is the easiest way to collect 2 to 3 pilot demonstrations.

- press `n` to solve motion to the dragged pose
- press `g` to toggle the gripper
- press `c` to save the current trajectory and start a new one
- press `q` to quit
- press `h` for help

For a 2 to 3 demo pilot, start with `PickCube-v1`. `PushCube-v1` is also reasonable. `StackCube-v1` is higher variance and should be treated as a harder target.

## Official Baseline Coverage

This suite is built around ManiSkill's documented baseline families:

- RL: PPO, SAC
- IL: BC, ACT, Diffusion Policy
- demo-aware online learning: RFCL, RLPD

Those official baselines live in the ManiSkill repository under `examples/baselines/`. The bootstrap script clones that repository into `maniskill_suite/external/ManiSkill` so the generated commands have a stable local target.

## Runtime Notes

- `tensorboard` is required by the official PPO, SAC, and BC entrypoints and is included in `requirements.txt`.
- The workspace shell exports `PYTHONPATH` to user-site packages, so the suite scripts explicitly clear `PYTHONPATH` before invoking the local `.venv` interpreter.
- Official RL demo downloads use control-mode-specific files such as `trajectory.none.pd_joint_delta_pos.physx_cuda.h5`; the suite scripts now resolve those automatically.
- RFCL and RLPD required extra JAX dependencies beyond the base install and, in this workspace, currently fall back to CPU because the installed `jaxlib` is not CUDA-enabled.
- ACT and Diffusion Policy need extra optional packages beyond the base suite: `torchvision` for ACT and the `diffusers` stack for Diffusion Policy.
- Current validated pilot results live in `reports/experiment_results.md` and in the per-combination folders for `pickcube_bc`, `pickcube_sac`, `pickcube_rfcl`, and `pickcube_rlpd`.