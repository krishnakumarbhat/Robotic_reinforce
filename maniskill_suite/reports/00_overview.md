# Overview

This suite translates your research direction into a practical ManiSkill-first experiment program.

The core thesis is simple:

- imitation learning gives you fast policy initialization from demonstrations
- reinforcement learning gives you robustness, recovery, and the chance to exceed the demonstrator
- simulation is the only place where this can be explored cheaply enough to matter when hardware and GPU budget are tight

Your research note is focused on the failure mode of pure imitation learning, the inefficiency of pure reinforcement learning, and the need for a Real-Sim-Real loop. This repository cannot implement the full real-world deployment side on its own, but it can implement the simulated center of that loop in a disciplined way.

The design choices in this suite follow that logic.

## What The Matrix Tests

Three task families were selected.

- `PickCube-v1` for the cleanest grasping smoke test
- `PushCube-v1` for contact-rich but low-horizon manipulation
- `StackCube-v1` for a harder long-horizon manipulation target

Seven algorithm families were selected.

- PPO and SAC as reward-only RL baselines
- BC, ACT, and Diffusion Policy as pure imitation baselines
- RFCL and RLPD as demo-aware online methods that move closer to the IL-to-RL transition discussed in your report

The result is a matrix that is broad enough to be meaningful, but still realistic for a 4 GB VRAM machine if you stay disciplined and begin with state observations.

## Why The Suite Starts With State Observations

With 4 GB of VRAM, visual training is not the first problem to solve. The first problem is whether the control loop, evaluation protocol, dataset conversion, and baseline commands are all working. State-based policies are the fastest way to establish that.

That is why this suite treats RGB and RGBD as stretch work instead of default work.

## What Counts As A Useful Result

A useful result in this repository is not only a high success rate.

It can also be one of the following:

- proof that 2 to 3 teleop demos are too few for BC but sufficient to warm-start RFCL
- proof that SAC outperforms PPO on sample efficiency for a chosen control mode
- proof that controller choice during replay changes IL stability more than algorithm choice
- proof that `success_once` and `success_at_end` diverge strongly on stacking, which indicates recovery weakness

These are concrete experimental outcomes tied directly to the research framing you wrote.

## What This Suite Does Not Pretend To Solve

This is not yet a full sim-to-real deployment pipeline.

It does not model battery sag, camera distortion, latency, backlash, or real robot calibration errors. It gives you the structure needed to compare demonstration-aware algorithms in simulation, record the right metrics, and identify which methods are worth carrying forward.

That is the right first move on a 4 GB machine.