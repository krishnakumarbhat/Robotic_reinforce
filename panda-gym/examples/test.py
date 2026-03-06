import gymnasium as gym  # <-- Uses the new library
import time
import panda_gym 
from stable_baselines3 import SAC

# --- CONFIGURATION ---
ALGORITHM = SAC
ENV_ID = "PandaPush-v3" 
MODEL_PATH = "sac_pandapush-v3.zip"
# --- END CONFIGURATION ---

print(f"Creating environment: {ENV_ID}")
# We must create the env first for HerReplayBuffer
env = gym.make(ENV_ID, render_mode="human")

print(f"Loading model from: {MODEL_PATH}")
model = ALGORITHM.load(MODEL_PATH, env=env)
print("Model loaded successfully.")

print(f"Running test on {ENV_ID}...")
obs, info = env.reset() # gymnasium.reset() returns (obs, info)
try:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        
        # gymnasium.step() returns 5 values
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- ADDED THIS LINE ---
        # Slows down the simulation to 30 FPS so you can see it
        time.sleep(1/30)
        # ---------------------
        
        # 'done' is now 'terminated' or 'truncated'
        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, info = env.reset()
            
except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
except Exception as e:
    # This will now only catch if the window is closed
    if "Window" in str(e) or "pybullet" in str(e):
        print("Simulation window closed.")
    else:
        # Show any other unexpected errors
        print(f"An unexpected error occurred: {e}")
finally:
    env.close()
    print("Environment closed.")