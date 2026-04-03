# Mouse And Cursor IL Workflow

This is the shortest path from your own demonstrations to a real ManiSkill imitation baseline.

## Best First Task

Start with `PickCube-v1`.

If you want a second task, use `PushCube-v1`.

Only use `StackCube-v1` after the first two are working.

## Step 1: Collect 2 To 3 Teleop Demos

```bash
cd maniskill_suite
source .venv/bin/activate
./scripts/collect_teleop_demos.sh PickCube-v1
```

Use the click-and-drag interface.

- drag the end effector to a goal pose
- press `n` to motion plan
- press `g` to open or close the gripper
- press `c` after each successful trajectory
- press `q` when you have 2 to 3 good demonstrations

## Step 2: Replay To State Observations

```bash
./scripts/replay_demos.sh PickCube-v1 teleop state pd_ee_delta_pos 3 physx_cpu
```

This converts the compressed teleop trajectory into the state dataset used by BC, ACT, and Diffusion Policy.

## Step 3: Train A Pure IL Baseline

For the fastest check, start with BC.

```bash
python scripts/run_from_matrix.py --combo pickcube_bc
```

Then compare against ACT.

```bash
python scripts/run_from_matrix.py --combo pickcube_act
```

Then compare against Diffusion Policy.

```bash
python scripts/run_from_matrix.py --combo pickcube_diffusion_policy
```

## Step 4: Try The Few-Demo Online Method

If 2 to 3 demos are not enough, collect 2 more and move to RFCL.

```bash
./scripts/replay_demos.sh PickCube-v1 teleop state pd_joint_delta_pos 5 physx_cpu
python scripts/run_from_matrix.py --combo pickcube_rfcl
```

## What To Expect

- BC may overfit and fail to recover from small state drift.
- ACT may be steadier than BC when the demos vary in timing.
- Diffusion Policy is the strongest pure IL candidate, but slower.
- RFCL is the best official baseline in this suite for tiny demo counts.

## Minimal Honest Goal

With only 2 to 3 mouse demos, the honest target is a functioning pilot pipeline, not a breakthrough result. The breakthrough work starts when you compare tiny-demo RFCL against BC and then increase the demo count only where it matters.