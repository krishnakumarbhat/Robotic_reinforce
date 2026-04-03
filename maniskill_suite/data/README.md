# Data Layout

This folder is for local ManiSkill assets, demonstrations, converted trajectories, and recorded videos.

Recommended structure:

- `raw/` for downloaded or teleoperated trajectories
- `processed/` for replayed state or RGB trajectories used by baselines
- `videos/` for saved rollouts and teleop captures

These large outputs are ignored by Git through the top-level `.gitignore`.