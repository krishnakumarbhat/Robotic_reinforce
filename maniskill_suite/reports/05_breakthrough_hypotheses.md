# Breakthrough Hypotheses

These are the highest-value ideas to test with this suite before expanding the compute budget.

## Hypothesis 1

Five high-quality teleop demos plus RFCL will outperform thirty low-quality teleop demos plus BC on `PickCube-v1`.

Why it matters:

- this directly tests whether an IL-to-RL bridge is better than brute-force demo count under limited human data

## Hypothesis 2

Replay controller choice will matter more than the pure IL algorithm on `PushCube-v1`.

Why it matters:

- if replayed actions are mismatched to the controller used in training, all IL methods can be handicapped before learning even starts

## Hypothesis 3

`success_once` and `success_at_end` will diverge on `StackCube-v1`, revealing fragile policies that can touch success but not stabilize it.

Why it matters:

- this exposes the difference between reaching a goal and maintaining a solved state, which is central for real deployment

## Hypothesis 4

State-based results will be sufficient to rank the useful algorithm families before any RGB work is justified on 4 GB VRAM.

Why it matters:

- you should only spend vision budget after the control-side ranking is clear

## Hypothesis 5

Official large prior datasets will make RLPD competitive on `PickCube-v1`, but tiny teleop data will not.

Why it matters:

- this separates few-demo methods from large-prior-data methods instead of mixing them unfairly