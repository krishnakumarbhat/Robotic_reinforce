# Algorithm Mapping

| Algorithm | Family | Role In Research Pipeline | Best Use In This Suite | Main Risk |
| --- | --- | --- | --- | --- |
| PPO | Online RL | Reward-only control baseline | Stable reference on state observations | Weak sample efficiency on harder tasks |
| SAC | Online RL | Stronger continuous-control RL baseline | Best RL-only comparison under small compute | Can still need careful tuning on stacking |
| BC | Pure IL | Fastest imitation baseline | Sanity check for demo quality and covariate shift | Collapses quickly with 2 to 3 noisy demos |
| ACT | Pure IL | Sequence-aware imitation baseline | Better than BC when actions are chunked or multi-modal | Heavier than BC and harder to debug |
| Diffusion Policy | Pure IL | Robust imitation of multi-modal human behavior | Best pure-IL candidate for teleop demos | Slower and more memory-sensitive |
| RFCL | Online demos | Few-demo IL-to-RL bridge | Best official candidate for 5-demo pilots | Extra JAX stack and more complex configs |
| RLPD | Offline-to-online RL | Prior-data acceleration of RL | Best when you have larger official demo sets | Poor fit for only 2 to 3 teleop demos |

## Practical Interpretation

- If you want the fastest truthful baseline, start with SAC.
- If you want to use your own mouse-collected demos immediately, start with BC and RFCL.
- If you want a stronger pure imitation result on human data, use Diffusion Policy after BC works.
- If you want to test the strongest claim from your research note, compare BC versus RFCL on the same tiny teleop dataset.