"""3 arm demos: KUKA iiwa + scrub-disc tool executes our top-3 measured paths.
1 trochoid (B 20/20 champion) / 2 orbit spiral (A 0.21->0.99 yaw-invariant) /
3 rounded rows (C1-smooth baseline). pybullet DIRECT + TinyRenderer, 640x480.
-> ~/Desktop/aegis_arm_{trochoid,orbit,rounded}.mp4
Run: ~/.venvs/infer/bin/python experiments/arm_demos_render.py [mode]
"""
import math
import os
import subprocess
import sys

import pybullet as p
import pybullet_data

OUT = os.path.expanduser("~/Desktop")
FPS, W, H = 24, 640, 480
R, CELL = 0.015, 0.05
HALF, SIDE = 0.20, 0.24


def build_paths():
    rows = [-SIDE / 2 + (r + 0.5) * CELL for r in range(max(1, int(SIDE / CELL)))]
    t1, PITCH = [], 0.02
    for r, v in enumerate(rows):
        n_loops = max(1, int(2 * HALF / PITCH))
        for i in range(n_loops * 24 + 1):
            th = 2 * math.pi * i / 24
            u = (-HALF + 2 * HALF * i / (n_loops * 24)) if r % 2 == 0 else \
                (HALF - 2 * HALF * i / (n_loops * 24))
            t1.append((u + R * math.cos(th), v + R * math.sin(th)))
    t2, Rd = [], math.hypot(HALF, SIDE / 2)
    n_turns = int(Rd / CELL) + 1
    for i in range(n_turns * 48 + 1):
        th = 2 * math.pi * i / 48
        rr = Rd * i / (n_turns * 48)
        t2.append((rr * math.cos(th), rr * math.sin(th)))
    t3 = []
    for r, v in enumerate(rows):
        us = [(-HALF + i * CELL / 2) for i in range(int(2 * HALF / (CELL / 2)) + 1)]
        if r % 2:
            us = us[::-1]
        t3 += [(u, v) for u in us]
    return {"trochoid": t1, "orbit": t2, "rounded": t3}


TITLES = {
    "trochoid": "1 CHAMPION trochoid - 20/20 (p=0.0083)",
    "orbit": "2 ORBIT spiral - A 0.21 -> 0.99 yaw-proof",
    "rounded": "3 ROUNDED rows - C1-smooth baseline",
}


def plt_imsave(path, arr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(path, arr)


def render(mode, pts):
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("table/table.urdf", [0, 0, 0])
    arm = p.loadURDF("kuka_iiwa/model.urdf", [0, -0.52, 0.63])
    plate = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[HALF, SIDE / 2, 0.008]),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[HALF, SIDE / 2, 0.008],
            rgbaColor=[0.15, 0.35, 0.8, 1]),
        basePosition=[0, 0.12, 0.638])
    PX, PY, PZ = 0, 0.12, 0.646
    tool = p.createMultiBody(baseMass=0.2,
                             baseCollisionShapeIndex=p.createCollisionShape(
                                 p.GEOM_CYLINDER, radius=0.035, height=0.02),
                             baseVisualShapeIndex=p.createVisualShape(
                                 p.GEOM_CYLINDER, radius=0.035, length=0.02,
                                 rgbaColor=[1, 0.65, 0.1, 1]),
                             basePosition=[0, 0, 1])
    ee = 6
    step = max(1, len(pts) // 220)
    wps = pts[::step]
    traj = []
    for (u, v) in wps:
        tgt = [PX + u, PY + v, PZ + 0.002]
        q = list(p.calculateInverseKinematics(arm, ee, tgt, maxNumIterations=60,
                                              residualThreshold=1e-4))
        traj.append((list(q), (u, v)))
    view = p.computeViewMatrix([1.05, -0.85, 1.1], [PX, PY + 0.05, 0.64],
                               [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(55, W / H, 0.05, 5.0)
    frames_dir = f"/tmp/arm_{mode}"
    os.makedirs(frames_dir, exist_ok=True)
    cur = traj[0][0]
    fi = 0
    for qi, (u, v) in traj:
        for a in range(4):
            qa = [c + (n - c) * (a + 1) / 4 for c, n in zip(cur, qi)]
            for j, ang in enumerate(qa):
                p.resetJointState(arm, j, ang)
            ee_pos = p.getLinkState(arm, ee)[0]
            p.resetBasePositionAndOrientation(
                tool, [ee_pos[0], ee_pos[1], ee_pos[2] - 0.035], [0, 0, 0, 1])
            p.stepSimulation()
            if fi % 2 == 0:
                _, _, rgb, _, _ = p.getCameraImage(W, H, view, proj,
                                                   renderer=p.ER_TINY_RENDERER)
                import numpy as np
                frame = np.array(rgb, dtype="uint8").reshape(H, W, 4)[:, :, :3]
                plt_imsave(f"{frames_dir}/f{fi // 2:04d}.png", frame)
            fi += 1
        cur = qi
    p.disconnect()
    mp4 = f"{OUT}/aegis_arm_{mode}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
                    "-i", f"{frames_dir}/f%04d.png",
                    "-vf", f"drawtext=text='{TITLES[mode]}':fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6:x=10:y=10",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", mp4],
                   check=True)
    print("wrote", mp4, f"{os.path.getsize(mp4) / 1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    modes = build_paths()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for m, pts in modes.items():
        if only and m != only:
            continue
        print("RENDER", m, len(pts), flush=True)
        render(m, pts)
